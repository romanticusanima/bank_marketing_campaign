import matplotlib.pyplot as plt
import seaborn as sns

def show_distribution(data, col):
    plt.figure(figsize=(8, 4))
    plt.title(f'Distribution of {col}')
    sns.histplot(data[col], color='g', kde=True, bins=30)  
    plt.show()


def show_kde(data, column):
    plt.figure(figsize=(8, 4))
    df0 = data[data['y'] == 'no']
    df1 = data[data['y'] == 'yes']
    sns.kdeplot(df0[column], label='Not Subscribed', cut=0)
    sns.kdeplot(df1[column], label='Subscribed', cut=0)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def get_max_value(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    max_value = (Q3 + 1.5 * IQR)
    return max_value


def bi_cat_countplot(df, column, hue_column):
    unique_hue_values = df[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(16,6)

    pltname = f'Normalized values distribution by category: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)
    ax = proportions.unstack(hue_column).sort_values(by=unique_hue_values[0], ascending=False).plot.bar(ax=axes[0], title=pltname)
    ax.margins(y=0.2)

    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%', rotation=90, fontsize=10, padding=2)

    pltname = f'Data amount by category: {column}'
    counts = df.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
        ).plot.bar(ax=axes[1], title=pltname)
    ax.margins(y=0.2)

    for container in ax.containers:
      ax.bar_label(container, rotation=90, fontsize=10, padding=2)


def bi_cat_distribution(df, column, hue_column):
    unique_hue_values = df[hue_column].unique()
    fig, ax = plt.subplots(figsize=(8,5))

    pltname = f'Normalized values distribution by category: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)

    ax = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
        ).plot.bar(ax=ax, title=pltname)
    ax.margins(y=0.2)

    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%', rotation=90, fontsize=10, padding=2)

    plt.tight_layout()
    plt.show()


def bi_cat_distribution_sort_index(df, column, hue_column):
    unique_hue_values = df[hue_column].unique()
    fig, ax = plt.subplots(figsize=(8,5))

    pltname = f'Normalized values distribution by category: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)

    ax = proportions.unstack(hue_column).sort_index().plot.bar(ax=ax, title=pltname)
    ax.margins(y=0.2)

    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%', rotation=90, fontsize=10, padding=2)

    plt.tight_layout()
    plt.show()