

class TfidfSentenceRetriever:

    def __init__(self):
        pass

    def score_sentences(self, claim: str, sentences: list) -> list:
        """

        Return the TF-IDF similarity between each sentence in the list of input sentences and the claim

        Args:
            claim: the text containing the claim
            sentences: the list of all the sentences retrieved from all the predicted documents from the previous stage.
                        Sentences come from different documents but they have all been added to this single list

        Returns: a list where the ith element is the score for the corresponding ith sentence in the input sentences

        Example:

        sentences: ["sentence_1", "sentence_2", ..., "sentence_n"]
        return: [0.12, 0.71, ..., 0.01] (0.12 is the score for sentence_1, and 0.71 is the score for sentence_2, etc.)

        """
        # TODO implement
        return [0.0] * len(sentences)
