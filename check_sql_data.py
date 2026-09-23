import sqlite3

DB_PATH = "Whatsaapp_to_Sql\\bond_deals.db"  # update path if needed

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row  # lets us access columns by name

rows = conn.execute("SELECT * FROM bond_deals").fetchall()

if not rows:
    print("No records found.")
else:
    columns = rows[0].keys()
    print(" | ".join(columns))
    print("-" * 80)
    for row in rows:
        print(" | ".join(str(row[c]) for c in columns))

conn.close()