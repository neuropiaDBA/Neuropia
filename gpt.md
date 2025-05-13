좋습니다! 업로드한 \*\*타이타닉호 데이터(titanic1309.csv)\*\*는 총 **1,309명**의 승객 정보를 담고 있고, 12개의 열(column)로 구성되어 있습니다. 팀플 발표용 분석을 위해 다음과 같은 순서로 주피터 노트북을 구성하는 것을 추천드립니다.

---

## 🎯 분석 과제 구조 제안 (Jupyter Notebook 목차)

### 🧾 1. 프로젝트 개요

* 과제 목적: 타이타닉호 생존자 분석
* 데이터 출처 및 소개
* 사용 도구: Python (Pandas, Matplotlib, Seaborn)

---

### 📊 2. 데이터 확인 및 전처리

```python
# 데이터 불러오기
import pandas as pd
df = pd.read_csv('titanic1309.csv')

# 기본 정보 확인
df.info()
df.describe()
df.head()
```

#### 결측치 확인 및 처리

```python
# 결측치 확인
df.isnull().sum()

# 결측치 처리 예시
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df.drop(columns=['Cabin'], inplace=True)  # Cabin은 결측치가 많아 제거
```

---

### 📈 3. EDA (탐색적 데이터 분석)

#### 생존자 비율

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()
```

#### 성별에 따른 생존율

```python
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.show()
```

#### 객실 등급(Pclass)에 따른 생존율

```python
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Class')
plt.show()
```

#### 나이에 따른 생존 분석

```python
sns.histplot(data=df, x='Age', hue='Survived', bins=20, kde=True)
plt.title('Age Distribution by Survival')
plt.show()
```

#### 가족 동반자 수 (SibSp + Parch)와 생존율

```python
df['FamilySize'] = df['SibSp'] + df['Parch']
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title('Survival Rate by Family Size')
plt.show()
```

---

### 🧠 4. 주요 인사이트 정리

예시:

* **여성의 생존율이 높다.**
* **1등실 승객이 다른 클래스보다 생존율이 높다.**
* **아이(나이 어린 승객)의 생존율이 다소 높은 경향.**
* **가족 단위 승객은 생존율이 높을 수 있다.**

---

### 🗣️ 5. 발표 및 질의응답용 예상 질문

예:

* Q. 어떤 기준으로 결측치를 처리했나요?
* Q. 분석 과정에서 가장 흥미로웠던 점은 무엇인가요?

---

이제 위 구조에 맞춰 주피터 노트북을 하나하나 만들어드릴게요.
**우선 EDA 파트부터 주피터노트북 코드로 정리해드릴까요?**
혹은 **전체 발표 노트북**을 한 번에 구성해드릴까요?
