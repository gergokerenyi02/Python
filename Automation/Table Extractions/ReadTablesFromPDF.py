import camelot

# Read the PDF file
tables = camelot.read_pdf('test.pdf', pages='1', flavor='lattice')

tables.export('test.csv', f='csv', compress=False)

#tables[0].to_csv('test.csv')
