from orator.migrations import Migration


class CreateNounsTable(Migration):

    def up(self):
        """
        Run the migrations.
        """
        with self.schema.create('nouns') as table:
            table.increments('id')
            table.long_text('text')
            table.long_text('doc_id')
            table.timestamps()

    def down(self):
        """
        Revert the migrations.
        """
        pass
