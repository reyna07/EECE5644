import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,confusion_matrix,roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt


X = []
y = [] 


X = pd.read_csv('X_snts.csv') #read the data file
y = pd.read_csv('y_snts.csv')

y = y.values.ravel() #transpose the y matrix

alpha_values = np.logspace(-15, 5, base=2) #range of alpha

accuracies = []

for i in range(10): #repeating for 10 times
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    accuracies_for_iteration = []

    
    for alpha in alpha_values:
        
        classifr = MultinomialNB(alpha=alpha)
        classifr.fit(X_train, y_train) #fitting
        accuracy = classifr.score(X_test, y_test) #directly testing
        accuracies_for_iteration.append(accuracy)

    accuracies.append(accuracies_for_iteration)

accuracies = np.array(accuracies)
avg_accuracy = np.mean(accuracies, axis=0) #taking the mean
std_deviation = np.std(accuracies, axis=0) #taking the standard deviation


best_alpha_index = np.argmax(avg_accuracy) #best accuracy index
best_alpha = alpha_values[best_alpha_index] #best alpha


print(f"Best alpha value: {best_alpha}")
print(f"Maximum average accuracy: {avg_accuracy[best_alpha_index]}")


plt.errorbar(alpha_values, avg_accuracy, yerr=std_deviation, fmt='-o', capsize=5) #error plot for std deviation and mean
plt.xlabel('Alpha')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy as a Function of Alpha')
plt.xscale('log')
plt.grid(True)
plt.show()


best_classifr = MultinomialNB(alpha=best_alpha) #best classifier
best_classifr.fit(X, y)
best_features = best_classifr.feature_log_prob_

base_class = np.argsort(best_features[4])[-5:] #sorting the labels
cont_class = np.argsort(best_features[3])[-5:]
own_class = np.argsort(best_features[2])[-5:]
aim_class = np.argsort(best_features[1])[-5:]
misc_class = np.argsort(best_features[0])[-5:]


feature_names = ['the', 'minimum'   , 'description'   , 'length'   , 'principle'   , 'for'   , 'online'   , 'sequence'   , 'estimation/prediction'   , 'in'   , 'a'   , 'proper'   , 'learning'   , 
                 'setup'   , 'is'   , 'studied'   , 'if'   , 'underlying'   , 'model'   , 'class'   , 'discrete'   , 'then'   , 'total'   , 'expected'   , 'square'   , 'loss'   , 'particularly'   , 
                 'interesting'   , 'performance'   , 'measure'   , 'this'   , 'quantity'   , 'finitely'   , 'bounded'   , 'implying'   , 'convergence'   , 'with'   , 'probability'   , 'one'   , 'and'   , 'b'   , 
                 'it'   , 'additionally'   , 'specifies'   , 'speed'   , 'mdl'   , 'general'   , 'can'   , 'only'   , 'have'   , 'bounds'   , 'which'   , 'are'   , 'finite'   , 'but'   , 'exponentially'   , 'larger'   , 
                 'than'   , 'those'   , 'bayes'   , 'mixtures'   , 'we'   , 'show'   , 'that'   , 'even'   , 'case'   , 'contains'   , 'bernoulli'   , 'distributions'   , 'derive'   , 'new'   , 'upper'   , 'bound'   , 'on'   , 
                 'prediction'   , 'error'   , 'countable'   , 'classes'   , 'implies'   , 'small'   , 'comparable'   , 'to'   , 'certain'   , 'important'   , 'discuss'   , 'application'   , 'machine'   , 'tasks'   , 'such'   , 'as'   , 
                 'classification'   , 'hypothesis'   , 'testing'   , 'generalization'   , 'of'   , 'iid'   , 'models'   , 'mixture'   , 'solomonoff'   , 'induction'   , 'marginalization'   , 'all'   , 'these'   , 'terms'   , 'refer'   ,
                   'central'   , 'obtain'   , 'predictive'   , 'distribution'   , 'by'   , 'integrating'   , 'product'   , 'prior'   , 'evidence'   , 'over'   , 'many'   , 'cases'   , 'however'   , 'computationally'   , 'infeasible'   , 
                   'sophisticated'   , 'approximation'   , 'expensive'   , 'or'   , 'map'   , 'maximum'   , 'posteriori'   , 'estimator'   , 'both'   , 'common'   , 'its'   , 'own'   , 'sake'   , 'use'   , 'largest'   , 'practice'   , 'usually'   , 
                   'being'   , 'approximated'   , 'too'   , 'since'   , 'local'   , 'determined'   , 'how'   , 'good'   , 'predictions'   , 'question'   , 'has'   , 'attracted'   , 'much'   , 'attention'   , 'an'   , 'quality'   , 'cumulative'   , 'predictor'   , 'particular'   , 'often'   , 'considered'   , 'assume'   , 'outcome'   , 'space'   , 'continuously'   , 'parameterized'   , 'unbounded'   , 'growing'   , 'symbol'   , ''   , 'where'   , 'sample'   , 'size'   , 'citation'   , 'corresponds'   , 'instantaneous'   , 'losses'  , 'behave'  , 'similarly'  , 
                   'under'  , 'appropriate'  , 'conditions'  , 'specific'  , 'note'  , 'order'  , 'do'  , 'continuous'  , 'needs'  , 'discretize'  , 'parameter'  , 'see'  , 'also'  , 'other'  , 'hand'  , "solomonoff's"  , 'theorem'  , 'namely'  , 'weight'  , 'true'  , 'necessary'  , 'assumption'  , 'contained'  , 'ie'  , 'dealing'  , 'been'  , 'demonstrated'  , 'be'  , 'essential'  , 'violated'  , 'may'  , 'fail'  , 'very'  , 'badly'  , 'shown'  , 'holds'  , 'sharp'  , 'probabilities'  , 'contrast'  , 'gives'  , 'sense'  , 'errors'  , 'magnitude'  , 
                   'cannot'  , 'occur'  , 'so'  , 'while'  , 'worse'  , 'compared'  , 'avoid'  , 'term'  , 'rate'  , 'here'  , 'identical'  , 'eg'  , 'monotonically'  ,
                     'decreasing'  , 'not'  , 'necessarily'  , 'therefore'  , 'natural'  , 'ask'  , 'there'  , 'present'  , 'work'  , 'concentrate'  , 'simplest'  , 'possible'  , 'stochastic'  , 'just'  , 'becomes'  , 'estimates'  , 'directly'  , 'uses'  , 'nevertheless'  , 'consistency'  , 'terminology'  , 'keep'  , 'might'  , 'surprising'  , 'discover'  , 'still'  , 'exponential'  , 'will'  , 'give'  , 'mild'  , 'guaranteeing'  , 'moreover'  , 'well-known'  , 'likelihood'  , 'decays'  , 'same'  , 'measured'  , 'more'  , 'statements'  , 'briefly'  , 
                   'discussed'  , 'section'  , 'motivation'  , 'consider'  , 'arises'  , 'algorithmic'  , 'information'  , 'theory'  , 'from'  , 'computational'  , 'point'  , 'view'  , 'relevant'  , 'computable'  , 'some'  , 'fixed'  , 'universal'  , 'turing'  , 'precisely'  , 'prefix'  , 'thus'  , 'each'  , 'program'  , 'countably'  , 'programs'  , 'they'  , 'semimeasures'  , 'strings'  , 'need'  , 'halt'  , 'otherwise'  , 'were'  , 'measures'  , 'corresponding'  , 'agree'  , 'binary'  , 'defined'  , 'two'  , 'negative'  , 'kraft'  , 'inequality'  , 
                   'priors'  , 'sum'  , 'up'  , 'at'  , 'most'  , 'call'  , 'given'  , 'related'  , 'isomorphic'  , 'set'  , 'reals'  , 'shortest'  , 'string'  , 'generated'  , 'high'  , 'two-part'  , 'complexity'  , 'respect'  , 'does'  , 'exceed'  , 'constant'  , 'vovk'  , 'save'  , 'additive'  , 'reduced'  , 'example'  , 'task'  , 'classifying'  , 'instance'  , 'after'  , 'having'  , 'seen'  , 'instance,class'  , 'pairs'  , 'phrased'  , 'predict'  , 'continuation'  , 'typically'  , 'generalize'  , 'conditionalized'  , 'inputs'  , 'solve'  , 'problems'  , 
                   'standard'  , 'form'  , 'obvious'  , 'proofs'  , 'paper'  , 'our'  , 'main'  , 'tool'  , 'obtaining'  , 'results'  , 'kullback-leibler'  , 'divergence'  , 'lemmata'  , 'stated'  , 'shows'  , 'obtained'  , 'latter'  , 'weights'  , 'subject'  , 'treats'  , 'finally'  , 'conclusions'  , 'although'  , 'internet'  , 'level'  , 'topology'  , 'extensively'  , 'past'  , 'few'  , 'years'  , 'little'  , 'known'  , 'about'  , 'details'  , 'taxonomy'  , 'node'  , 'represent'  , 'wide'  , 'variety'  , 'organizations'  , 'e'  , 'g'  , 'large'  , 'isp'  ,
                     'private'  , 'business'  , 'university'  , 'vastly'  , 'different'  , 'network'  , 'characteristics'  , 'external'  , 'connectivity'  , 'patterns'  , 'growth'  , 'tendencies'  , 'properties'  , 'hardly'  , 'neglect'  , 'working'  , 'veracious'  , 'representations'  , 'simulation'  , 'environments'  , 'introduce'  , 'radically'  , 'approach'  , 'based'  , 'techniques'  , 'ases'  , 'into'  , 'successfully'  , 'classify'  , 'number'  , 'percent'  , 'accuracy'  , 'release'  , 'community'  , 'dataset'  , 'augmented'  , 'attributes'  , 
                   'used'  , 'believe'  , 'serve'  , 'invaluable'  , 'addition'  , 'further'  , 'understanding'  , 'structure'  , 'evolution'  , 'rapid'  , 'expansion'  , 'last'  , 'decades'  , 'produced'  , 'scale'  , 'system'  , 'thousands'  , 'diverse'  , 'independently'  , 'managed'  , 'networks'  , 'collectively'  , 'provide'  , 'global'  , 'across'  , 'spectrum'  , 'geopolitical'  , 'globally'  , 'routable'  , 'identifiers'  , 'increased'  , 'less'  , 'exerting'  , 'significant'  , 'pressure'  , 'interdomain'  , 'routing'  , 'well'  , 'functional'  , 
                   'structural'  , 'parts'  , 'impressive'  , 'resulted'  , 'heterogenous'  , 'highly'  , 'complex'  , 'challenges'  , 'accurate'  , 'realistic'  , 'modeling'  , 'infrastructure'  , 'intermix'  , 'owned'  , 'operated'  , 'backbone'  , 'providers'  , 'regional'  , 'access'  , 'universities'  , 'companies'  , 'statistical'  , 'faithfully'  , 'characterizes'  , 'types'  , 'critical'  , 'path'  , 'toward'  , 'knowledge'  , 'mandatory'  , 'augmenting'  , 'synthetically'  , 'constructed'  , 'topologies'  , 'intra'  , 'inter'  , 'router'  , 'expect'  , 
                   'dual'  , 'homed'  , 'drastically'  , 'company'  , 'likely'  , 'contain'  , 'dozens'  , 'internal'  , 'routers'  , 'hosts'  , 'elements'  , 'switches'  , 'servers'  , 'firewalls'  , 'probably'  , 'single'  , 'simple'  , 'diversity'  , 'among'  , 'accurately'  , 'augment'  , 'misc'  , 'characterize'  , 'composing'  , 'annotating'  , 'their'  , 'prerequisite'  , 'exhibit'  , 'service'  , 'grow'  , 'attracting'  , 'customers'  , 'engaging'  , 'agreements'  , 'isps'  , 'connect'  , 'through'  , 'significantly'  , 'time'  , 'categorizing'  , 'identify'  , 
                   'develop'  , 'mapping'  , 'ip'  , 'addresses'  , 'users'  , 'traffic'  , 'analysis'  , 'studies'  , 'required'  , 'distinguish'  , 'between'  , 'packets'  , 'come'  , 'home'  , 'realize'  , 'goal'  , 'checking'  , 'type'  , 'originates'  , 'address'  , 'lies'  , 'construct'  , 'representative'  , 'algorithm'  , 'empirically'  , 'observed' 
]


print("Top 5 words for the AIMX class:")
for i in aim_class:
    print(feature_names[i])
print("Top 5 words for the OWNX class:")
for i in own_class:
    print(feature_names[i])
print("Top 5 words for the CONTRAST class:")
for i in cont_class:
    print(feature_names[i])
print("Top 5 words for the BASE class:")
for i in base_class:
    print(feature_names[i])
print("Top 5 words for the MISC class:")
for i in misc_class:
    print(feature_names[i])

