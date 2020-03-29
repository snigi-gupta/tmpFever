from torchtext.data import Example
import six

class FeverExample:

    @classmethod
    def fromlist(cls, data, fields):
        ex = Example()
        for (name, field), val in zip(fields, data):
            if field is not None:
                if isinstance(val, six.string_types):
                    val = val.rstrip('\n')
                # Handle field tuples
                if isinstance(name, tuple):
                    for n, f in zip(name, field):
                        setattr(ex, n, f.preprocess(val))
                else:
                    if name == "evidence":
                        for i in range(len(val)):
                            for j in range(len(val[i])):
                                val[i][j] = field.preprocess(val[i][j])
                        setattr(ex, name, val)
                    else:
                        res = field.preprocess(val)
                        setattr(ex, name, res)
        return ex

