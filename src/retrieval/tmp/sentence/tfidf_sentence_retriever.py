

class TfidfSentenceRetriever:

    def __init__(self):
        pass

    def score_sentences(self, claim: str, sentences: list) -> list:
        """

        Return the TF-IDF similarity between each sentence in the list of input sentences and the claim

        Args:
            claim: the text containing the claim
            sentences: the list of sentences retrieved from multiple documents

        Returns: a list where the ith element is the score for the corresponding ith sentence in the input sentences

        """
        return [0] * len(sentences)
