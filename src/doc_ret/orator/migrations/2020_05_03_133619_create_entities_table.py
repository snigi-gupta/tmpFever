from orator.migrations import Migration


class CreateEntitiesTable(Migration):

    def up(self):
        """
        Run the migrations.
        """
        with self.schema.create('entities') as table:
            table.increments('id')
            table.long_text('type')
            table.long_text('text')
            table.long_text('doc_id')
            table.timestamps()

    def down(self):
        """
        Revert the migrations.
        """
        pass
