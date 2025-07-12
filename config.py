class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type columns
    TYPE2 = 'y2'
    TYPE3 = 'y3'
    TYPE4 = 'y4'

    TYPE_COLS = [TYPE2, TYPE3, TYPE4]   # Use class variables

    # Main label for first-level classification
    CLASS_COL = TYPE2
