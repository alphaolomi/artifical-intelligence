from sklearn.datasets import load_digits
from sklearn.model_selection imprt train_tes_aplit
import nump as mp
import matplotlib.pyplot as pyplot
import seaborn as sns
from sklearn import metrics

%matplotlib inline

digits = load_digits()

print("Image Data Shape ", digits.data.shape)
print("Label Data Shape", digits.target.shape)



import nump as mp
import matplotlib.pyplot as pyplot
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate( zip(digits.data[0:5], digits.target[0:5]) ):
    plt.subplot( 1 ,5 index + 1 )
    plt.imshow( np.reshape( image, (8,8) ), cmap = plt.cm.gray )
    plt.title('Train %i\n' % label , fontsize = 20)

from sklearn.model_selection imprt train_tes_split
x_train, x_test, y_train, y_test = train_tes_split(digits.data, digits.target,test_size= 0.23, random_state=2)

print(x_train.shape)
print(y_train.shape)


print(x_test.shape)
print(y_test.shape)



from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train, y_train)

# Retuen a Numpy Array
# predict for One Observation (image)

print(regressor.predict(x_test[0].reshape(1,-1)))

# doig more predictions
regressor.predict(x_test[0:10])

# Getting all the predictions
predictions = regressor.predict(x_test)


# Evalute the model
score = regressor.score(x_test, y_test)
print(score)



# Make a confussion matrix
from sklearn import metrics
conf_matrix = metrics.confussion_matrix(y_test, predictions)
print(conf_matrix)


# Plot the predictions
plt.figure( figsize= (9,9))
sns.heatmap(conf_matrix, annot=True, fmt=".3f" , linewidths=.5, square = True, cmap="Blue_r")
plt.ylabel('Actual Label');
plt.xlabel('Predicted label');
all_smaple_title = ''.format(score)
plt.title(all_smaple_title, size= 15)


index = 0
classiedIndex = []
for predict,actual in zip(predictions, y_test):
    if predict== actual:
        classiedIndex.append(index)
    index +=
plt.figure( figsize=(20,3) )
for plotIndex, wrong in enumerate(classiedIndex[0:4]):
    plt.subplot(1,4,plotIndex+1)
    ptt.imshow(np.reshape(x_test[wrong], (8,8)), cmap=plt.cm.gray)
    plt.title("Predicted: {}, Actual : {}".format(predictions[wrong], y_test[wrong]),fontsize=20 )
