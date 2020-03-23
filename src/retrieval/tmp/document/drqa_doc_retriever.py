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

        Example:

        claim: "The Shining was a film made in 1980"
        k: 5
        returns: ['Shining_Star', 'Comrade_Artemio', 'Shining_Path', 'The_Shining_-LRB-film-RRB-', 'Shining_Through_-LRB-disambiguation-RRB-']

        """
        return self.ranker.closest_docs(claim, k)
