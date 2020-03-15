from drqa import retriever

class DrqaDocRetriever:

    def __init__(self, saved_model_path):
        self.ranker = retriever.get_class('tfidf')(tfidf_path=saved_model_path)

    def closest_docs(self, claim: str, k: int) -> list:
        """
        Return the closest k document to the claim

        Args:
            claim: the text containing the claim
            k: the number of document to return

        Returns: the k documents

        """
        return self.ranker.closest_docs(claim, k)
