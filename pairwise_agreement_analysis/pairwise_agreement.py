import combinations
import pandas as pd
#

data = pd.read_csv('ratings_data.csv')
def percentage_agreement(scores):
    """
    Calculate the percentage agreement among evaluators for a list of scores.
    Agreement is when two evaluators give the same score.
    """
    if len(scores) < 2:
        return 1.0  

    total_pairs = 0
    agreement_pairs = 0

    for a, b in combinations(scores, 2):
        total_pairs += 1
        if a == b:
            agreement_pairs += 1
    return agreement_pairs / total_pairs if total_pairs > 0 else 0.0

criteria = ['factuality',
       'appropriatness', 'adequacy', 'expert recall', 'self_awareness',
       'empathy', 'clinical reasoning', 'fluency/clarity', 'hallucination',
       'bias', 'harm']


variables = []
means = []
raws = []

for i in criteria:
    var1 = data[['answer_id', i]]
    var1.rename(columns={i:'score'}, inplace=True)
    var1['score'] = var1['score'].astype(int)
    g = var1.groupby('ans_id')['score'].apply(list)
    g_per = g.apply(percentage_agreement)
    m = g_per.mean()
    all_scores = [score for sublist in g for score in sublist]
    variables.append(i)
    means.append(m)
    raws.append(all_scores)

quest_scores = pd.DataFrame(columns=['variable','percentage_agreement'])
quest_scores['variable'] = variables
quest_scores['percentage_agreement'] = means
quest_scores['percentage_agreement'] = quest_scores['percentage_agreement'].apply(lambda x:round(x,3))
