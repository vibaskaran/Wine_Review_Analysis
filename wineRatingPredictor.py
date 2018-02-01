from flask import Flask, render_template, request
import numpy as np 
import pandas as pd 
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


app = Flask(__name__)
app.debug = True

init_data = pd.read_csv("winemag-data_first150k.csv")

wineData = init_data[init_data.duplicated('description', keep=False)]
wineData = wineData[['province','variety','country','price','points','region_1','winery']]
wineData = wineData.dropna(subset=['province','variety','country','price','points','region_1','winery'], how='any')
wineData.drop_duplicates(keep='first')

clf = DecisionTreeClassifier(random_state=10)

x_unsplit = wineData[['variety', 'price', 'winery', 'region_1']]
y_unsplit = wineData['points']
x_unsplit = pd.get_dummies(x_unsplit, columns=['variety', 'winery', 'region_1'])
X_train, X_test, y_train, y_test = train_test_split(x_unsplit, y_unsplit, random_state=1, train_size=0.90)

X_train_mean = X_train.fillna({"price": x_unsplit['price'].mean()})
X_test_mean = X_test.fillna({"price":x_unsplit['price'].mean()})
clf.fit(X_train_mean, y_train)

y_predictions = clf.predict(X_test_mean)

dt_acc = accuracy_score(y_test,y_predictions)

variety_category = sorted(wineData['variety'].unique().tolist())
winery_category = sorted(wineData['winery'].unique().tolist())
region_1_category = sorted(wineData['region_1'].unique().tolist())

columnList = X_test_mean.columns.tolist()

def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0
        
def predict_points(varietyVal, priceVal, wineryVal, region1Val):
    df1 = pd.DataFrame({'variety': varietyVal, 'price': priceVal, 'winery': wineryVal, 'region_1': region1Val}, index=[0])
    new_sample = pd.get_dummies(df1)
    add_missing_dummy_columns(new_sample, columnList)
    new_sample1_prd = clf.predict(new_sample)
    return new_sample1_prd[0]

wine150k = pd.read_csv("winemag-data_first150k.csv")
wine150k = wine150k.drop('Unnamed: 0',  axis=1)
wine130k = pd.read_csv("winemag-data-130k-v2.csv")
wine130k = wine130k.drop('Unnamed: 0',  axis=1)
df = pd.merge(wine130k, wine150k, how='outer', indicator=True)
onlyIn130kList_allCol = df[df['_merge']=='left_only'][wine130k.columns]
onlyIn130kList = onlyIn130kList_allCol[['variety', 'province', 'price', 'winery', 'region_1', 'points']]
uniqueIn130kList = onlyIn130kList[onlyIn130kList.duplicated(keep=False)]
uniqueIn130kList = uniqueIn130kList.drop_duplicates(keep='first')
uniqueIn130kList = uniqueIn130kList.dropna(subset=['province','variety','price','points','region_1'], how='any')
lastIndex = len(uniqueIn130kList)


def getDataBetweenRecrds(startIndex, endIndex):
    resRecords = uniqueIn130kList[startIndex:endIndex]
    return resRecords

def selectRandomWine():
    randomWineIndex = randint(1, lastIndex)
    selectedRecord = getDataBetweenRecrds(randomWineIndex-1,randomWineIndex)
    return selectedRecord

def isValPresent(valToCheck, lstToCheck):
    if valToCheck in lstToCheck:
        return True
    else:
        return False

def canWineBePredicted(variety,winery,region):
    varietyPresent = isValPresent(variety, variety_category)
    wineryPresent = isValPresent(winery, winery_category)
    region_1Present = isValPresent(region, region_1_category)
    if (varietyPresent and wineryPresent and region_1Present):
        return True
    else:
        return False

def selectAllDetails(selectedWine):
    lst1 = []
    onlyIn130kList_allCol.reset_index()
    for x in range(0,len(onlyIn130kList_allCol)):
        if(selectedWine['variety'].values[0] == onlyIn130kList_allCol[x:x+1]['variety'].values[0] and
           selectedWine['winery'].values[0] == onlyIn130kList_allCol[x:x+1]['winery'].values[0] and
           selectedWine['price'].values[0] == onlyIn130kList_allCol[x:x+1]['price'].values[0] and
           selectedWine['points'].values[0] == onlyIn130kList_allCol[x:x+1]['points'].values[0] and
           selectedWine['region_1'].values[0] == onlyIn130kList_allCol[x:x+1]['region_1'].values[0]):
            lst1.append(onlyIn130kList_allCol[x:x+1])
    return lst1[0]
    
def selectWineForPrediction(variety_category,winery_category,region_1_category):
    lstVal, retVal = [], []
    selectedWine = selectRandomWine()
    predictWine = canWineBePredicted(selectedWine['variety'].values[0],selectedWine['winery'].values[0],selectedWine['region_1'].values[0])
    if predictWine:
        selWine = selectAllDetails(selectedWine)
        lstVal.append(selWine)
        predictedVal = predict_points(selectedWine['variety'].values[0],selectedWine['price'].values[0],selectedWine['winery'].values[0],selectedWine['region_1'].values[0])
        lstVal.append(predictedVal)
        retVal = lstVal
    else:
        retVal = selectWineForPrediction(variety_category,winery_category,region_1_category)
    return retVal

def selectWine():
    selWine = selectWineForPrediction(variety_category,winery_category,region_1_category)
    return selWine

@app.route('/', methods=['GET'])
def dropdown():
    a = predictPoints()
    return a

@app.route("/predictPoints" , methods=['GET', 'POST'])
def predictPoints():
    selWine = selectWine()
    selectedWine = selWine[0]
    predictedRating = selWine[1]
    if int(selectedWine['points'].values[0]) >= int(predictedRating):
        if (int(selectedWine['points'].values[0]) - int(predictedRating)) == 0:
            predctdRange = str((predictedRating - 2)) + ' - ' + str((predictedRating + 2))
        elif (int(selectedWine['points'].values[0]) - int(predictedRating)) == 1:
            predctdRange = str((predictedRating - 1)) + ' - ' + str((predictedRating + 3))
        else:
            predctdRange = str((predictedRating)) + ' - ' + str((predictedRating + 4))
        # else if (int(selectedWine['points'].values[0]) - int(predictedRating)) == 2:
        #     predctdRange = str((predictedRating)) + ' - ' + str((predictedRating + 4))
        # else if (int(selectedWine['points'].values[0]) - int(predictedRating)) == 3:
        #     predctdRange = str((predictedRating + 1)) + ' - ' + str((predictedRating + 4))
        # else if (int(selectedWine['points'].values[0]) - int(predictedRating)) == 4:
        #     predctdRange = str((predictedRating)) + ' - ' + str((predictedRating + 4))
            
        # predctdRange = str(predictedRating) + ' - ' + str(selectedWine['points'].values[0])
    else:
        if (int(predictedRating) - int(selectedWine['points'].values[0])) == 1:
            predctdRange = str((int(selectedWine['points'].values[0]) - 1)) + ' - ' + str((int(selectedWine['points'].values[0]) + 3))
        else:
            predctdRange = str((int(selectedWine['points'].values[0]))) + ' - ' + str((int(selectedWine['points'].values[0]) + 4))
        

        # predctdRange = str(selectedWine['points'].values[0]) + ' - ' + str(predictedRating)

    if(request.form.get('varieties') is None and request.form.get('wineries') is None and request.form.get('regions') is None and request.form.get('price') is None):
        variety = variety_category[0]
        winery = winery_category[0]
        region = region_1_category[0]
        price = float(10)
    else:
        variety = request.form.get('varieties')
        winery = request.form.get('wineries')
        region = request.form.get('regions')
        price = float(request.form.get('price'))
    df1 = pd.DataFrame({
    'variety': variety,
    'winery': winery,
    'region_1': region,
    'price': price
    }, index=[0])

    # df1.style.set_properties(subset=['variety','winery','region_1','price'], **{'width': '300px'})
    predctdVal = predict_points(variety, price, winery, region)
    predctdRange2 = str((predctdVal - 2)) + ' - ' + str((predctdVal + 2))
    return render_template('index.html', varieties=variety_category, wineries=winery_category, regions=region_1_category, results2=selectedWine.to_html(index=False),results3=predictedRating, result=df1.to_html(index=False,col_space=10), result2=predctdVal, result3=predctdRange, result4=predctdRange2)


if __name__ == "__main__":
    app.run()