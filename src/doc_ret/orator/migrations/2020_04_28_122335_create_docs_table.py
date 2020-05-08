from orator.migrations import Migration


class CreateDocsTable(Migration):

    def up(self):
        """
        Run the migrations.
        """
        with self.schema.create('docs') as table:
            table.increments('id')
            table.long_text('page_id')
            table.long_text('text')
            table.long_text('lines')
            table.timestamps()

    def down(self):
        """
        Revert the migrations.
        """
        self.schema.drop('docs')
