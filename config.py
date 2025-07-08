class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Target/Type columns to test or predict
    TYPE2 = 'y2'
    TYPE3 = 'y3'
    TYPE4 = 'y4'

    TYPE_COLS = [y2, y3, y4]

    # If you have a main class or grouped label
    CLASS_COL = y2  # or whichever you want as main class
