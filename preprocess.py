import category_encoders as ce
import pandas as pd


data = pd.read_csv("employee_reviews.csv",delimiter=",")

# print(data)

encoder = ce.OrdinalEncoder(cols=[
								"summary",
								"company",
								"location",
								"dates",
								"job",
								"pros",
								"cons",
								"advice-to-mgmt"
								])



encoder.fit(data)

data_train = data[:27126]

data_val = data[27126:36169]

data_test = data[36169:]

X_train = encoder.transform(data_train)
X_val = encoder.transform(data_test)
X_test = encoder.transform(data_val)

X_train.to_csv("train.csv",index_label=False,index=False)
X_val.to_csv("val.csv",index_label=False,index=False)
X_test.to_csv("test.csv",index_label=False,index=False)


print(X_train.shape[0] + X_val.shape[0]+ X_test.shape[0])


