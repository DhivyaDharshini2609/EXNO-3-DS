## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:

STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)
<img width="379" height="251" alt="20" src="https://github.com/user-attachments/assets/daed7037-6bff-4690-8480-4a35e2992c1e" />

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Data.csv")

df=pd.DataFrame(data)

print(df)
<img width="660" height="251" alt="21" src="https://github.com/user-attachments/assets/f90a13dc-cd3a-45a4-ac1e-13a82d385593" />

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Data.csv")

df=pd.DataFrame(data)

print(df)
<img width="732" height="493" alt="22" src="https://github.com/user-attachments/assets/f4dde57d-997c-4128-9166-98ba75c48b1b" />

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold','Very Hot']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["Ord_1"]])

<img width="374" height="257" alt="23" src="https://github.com/user-attachments/assets/f25d4fa6-0b7d-48b9-a9b1-7f6329085eea" />

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)

<img width="503" height="487" alt="24" src="https://github.com/user-attachments/assets/a95053ac-00b0-4930-ab46-440e5b4cc786" />

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

<img width="486" height="648" alt="25" src="https://github.com/user-attachments/assets/9ecc8dec-5195-46d2-acb1-335ecb1dd276" />

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

<img width="486" height="630" alt="26" src="https://github.com/user-attachments/assets/2483d100-73dd-45d1-985d-02620733a446" />

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

df['bo2']=e1.fit_transform(df[["ord_2"]])

df

le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc

df['bo2']=e1.fit_transform(df[["ord_2"]])

df

<img width="1279" height="371" alt="27" src="https://github.com/user-attachments/assets/dfcd0a5b-f97d-4380-87c5-ef3013af6868" />

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

df['bo2']=e1.fit_transform(df[["ord_2"]])

df

le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse=False)

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

<img width="1334" height="736" alt="28" src="https://github.com/user-attachments/assets/3001aef1-0f87-4710-872b-b116f05f4025" />

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

df['bo2']=e1.fit_transform(df[["ord_2"]])

df

le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse=False)

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)

df2

<img width="1309" height="737" alt="29" src="https://github.com/user-attachments/assets/76303a70-2856-43f5-9683-88d59564251b" />

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

df['bo2']=e1.fit_transform(df[["ord_2"]])

df

le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse=False)

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)

df2

pd.get_dummies(df2,columns=["nom_0"])

<img width="1303" height="730" alt="30" src="https://github.com/user-attachments/assets/85e7057d-2dec-4bee-8aad-e47b38dc3305" />

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)

!pip install --upgrade category_encoders
<img width="1027" height="838" alt="31" src="https://github.com/user-attachments/assets/f0679bdf-cfb5-4726-ac27-bff64caeb2ca" />


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)

!pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

df=pd.read_csv("data.csv")

df


df=pd.read_csv("data.csv")

df=pd.DataFrame(data)

print(df)

be=BinaryEncoder()

nd=be.fit_transform(df['ord_2'])

dfb=pd.concat([df,nd],axis=1)

dfb1=df.copy()

dfb

<img width="621" height="621" alt="32" src="https://github.com/user-attachments/assets/e4e88353-83bf-40be-a239-e3d527ab2dc2" />

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)

!pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

df=pd.read_csv("data.csv")

df

be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

df

dfb=pd.concat([df,nd],axis=1)

dfb

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("Encoding Data.csv")

df=pd.DataFrame(data)

print(df)

from category_encoders import TargetEncoder

df=pd.read_csv("data.csv")

df

te=TargetEncoder()

CC=df.copy()

new=te.fit_transform(X=CC["City"],y=CC["Target"])

CC=pd.concat([CC,new],axis=1)

CC
<img width="971" height="793" alt="33" src="https://github.com/user-attachments/assets/17ce7f7c-cca4-4730-a31f-f1eaca58077a" />

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df


import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df

df.skew


import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df

np.log(df["Highly Positive Skew"])

<img width="731" height="631" alt="34" src="https://github.com/user-attachments/assets/f3cb75a2-230c-46a0-b3eb-95f0c5d83514" />

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df

np.reciprocal(df["Moderate Positive Skew"])


import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df

np.sqrt(df["Highly Positive Skew"])

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df

np.square(df["Highly Positive Skew"])
<img width="919" height="458" alt="35" src="https://github.com/user-attachments/assets/6da80c0a-ebfe-4224-98d1-0a32c2994a59" />


import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])

df


import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])

df

df.skew()


import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df
<img width="919" height="458" alt="35" src="https://github.com/user-attachments/assets/8c37c8c0-484e-4ca2-910e-39d4646924f5" />

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df
<img width="1347" height="604" alt="36" src="https://github.com/user-attachments/assets/71672042-b908-477d-b57c-58af27561fc1" />

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])

df
<img width="676" height="281" alt="37" src="https://github.com/user-attachments/assets/970ad1cc-c4b6-492a-a1f3-56e404af32ed" />

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df
<img width="657" height="284" alt="38" src="https://github.com/user-attachments/assets/32d0007b-09e2-41d2-99a1-1c16557ef6a4" />

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

<img width="651" height="265" alt="39" src="https://github.com/user-attachments/assets/8991ef38-b2c8-455b-9f21-c15b3cc48eb3" />

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df
<img width="1111" height="444" alt="40" src="https://github.com/user-attachments/assets/9bff7470-e8b8-4a65-a13c-b70ec9961856" />

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

plt.show()
<img width="519" height="155" alt="41" src="https://github.com/user-attachments/assets/2d52eb38-f5c5-4ff5-87a3-d2d63a5e7f38" />


import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df
<img width="1155" height="463" alt="42" src="https://github.com/user-attachments/assets/b3f3d241-159b-4f9f-9d27-baa619e395c4" />

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

<img width="781" height="570" alt="43" src="https://github.com/user-attachments/assets/dea99f71-3313-4a54-8edb-c1b7839f5301" />

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df
<img width="759" height="545" alt="44" src="https://github.com/user-attachments/assets/6dd37ed7-0673-4c5e-ac09-8099e6accc5a" />

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df["Highly Negative Skew"],line='45')

plt.show()
<img width="728" height="547" alt="45" src="https://github.com/user-attachments/assets/37920d96-5eba-4620-a334-d712d22e36c1" />

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("titanic_dataset.csv")

df
<img width="756" height="560" alt="46" src="https://github.com/user-attachments/assets/13dddaac-72ab-4bf5-8881-31170412ef72" />

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Age_1"]=qt.fit_transform(df[["Age"]])

sm.qqplot(df['Age'],line='45')

plt.show()
<img width="744" height="531" alt="47" src="https://github.com/user-attachments/assets/8163263d-cbe4-44fa-829d-8e860b7bf98f" />

RESULT:
Thus feature encoding and transformation process are performed.
# RESULT:
Thus the program to implement the linear regression using gradient descent is written and verified using 
Python programming.
       
