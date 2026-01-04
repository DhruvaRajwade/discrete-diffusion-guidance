from Bio import SeqIO

INPUT_FASTA = './UniDBP_1600_clean.fasta'
OUTPUT_FASTA = './UniDBP_1600_clean_final.fasta'

with open(OUTPUT_FASTA, 'w') as out_f:
    for record in SeqIO.parse(INPUT_FASTA, 'fasta'):
        # Extract Unique_ID from header
        try:
            unique_id = record.id.split('|')[1]
        except IndexError:
            print(f"Skipping malformed header: {record.id}")
            continue
        
        # Update the record ID and clear the description
        record.id = unique_id
        record.name = unique_id
        record.description = ''
        
        SeqIO.write(record, out_f, 'fasta')

print(f"Cleaned FASTA written to {OUTPUT_FASTA}")
