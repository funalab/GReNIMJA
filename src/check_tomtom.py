import pandas as pd


def check_prosite(dir):
    filename = dir + '/tomtom/tomtom.tsv'
    tomtom_result = pd.read_csv(filename, sep='\t')[:-3]
    Query_IDs = tomtom_result.drop_duplicates(subset='Query_ID')
    Target_IDs = Query_IDs.drop_duplicates(subset='Target_ID')['Target_ID']
    targets = []
    num = []

    for target in Target_IDs:
        targets.append(target)
        num.append(len(Query_IDs[Query_IDs['Target_ID'] == target]))
    data = []
    data.append(targets)
    data.append(num)
    print(data)
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(dir + '/tomtom/hit.txt', sep='\t', index=False, header=False)


def main():
    dir = 'a_motifs'
    check_prosite(dir)
    

if __name__ == '__main__':
    main()
