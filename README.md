# CryptoClustering 
Prepare the Data

    Use the StandardScaler() module from scikit-learn to normalize the data from the CSV file. 
    Create Scaled DataFrame: Set "coin_id" as the index for the new DataFrame.
crypto_data_scaled = StandardScaler().fit_transform(df_market_data[["price_change_percentage_24h","price_change_percentage_7d","price_change_percentage_14d","price_change_percentage_30d","price_change_percentage_60d","price_change_percentage_200d","price_change_percentage_1y"]])
crypto_data_scaled[:5]

    Create a DataFrame with the scaled data
crypto_data_scaled_df = pd.DataFrame(crypto_data_scaled, columns= ["price_change_percentage_24h","price_change_percentage_7d","price_change_percentage_14d","price_change_percentage_30d","price_change_percentage_60d","price_change_percentage_200d","price_change_percentage_1y"])
crypto_data_scaled_df.head()

    Copy the crypto names from the original DataFrame
crypto_data_scaled_df["coin_id"] =df_market_data.index

    Set the coin_id column as index
crypto_data_scaled_df = crypto_data_scaled_df.set_index("coin_id") 

    Display the scaled DataFrame
crypto_data_scaled_df.head()

![Code_d2XzLNnlND](https://github.com/user-attachments/assets/37c25862-8be0-4c47-b9a1-5e275a8a0375)




# Find the Best Value for k Using the Original Scaled DataFrame 
    Use the elbow method to find the best value for k using the following steps:
    # Create a list with the number of k-values from 1 to 11

 k = list(range(1,11)) 
 k

    # Create an empty list to store the inertia values
inertia = []
    
    # Create a for loop to compute the inertia with each possible value of k
    # Inside the loop:
    # 1. Create a KMeans model using the loop counter for the n_clusters
    # 2. Fit the model to the data using `df_market_data_scaled`
    # 3. Append the model.inertia_ to the inertia list 
for i in k:
    k_model = KMeans(n_clusters=i, random_state=0)
    k_model.fit(crypto_data_scaled_df)
    inertia.append(k_model.inertia_)
    
    # Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia}
 
    # Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data) 
df_elbow 

    # Plot a line chart with all the inertia values computed with
      the different values of k to visually identify the optimal value for k.
df_elbow.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)

![image](https://github.com/user-attachments/assets/42caf834-0d89-462c-b6a4-55e2e05499c6)


  
# Answer the following question in your notebook: What is the best value for k?
### The number four is the best value for K according to the elbow curve.


# Cluster Cryptocurrencies with K-means Using the Original Scaled Data

Use the following steps to cluster the cryptocurrencies for the best value for k on the original scaled data:

    # Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=1) 

    # Fit the K-Means model using the scaled DataFrame
model.fit(crypto_data_scaled_df)  

![image](https://github.com/user-attachments/assets/98e3c401-8f6c-4844-b690-a9e63cf2fcb7)

    # Predict the clusters to group the cryptocurrencies using the scaled DataFrame
k_4 = model.predict(crypto_data_scaled_df)

    # Print the resulting array of cluster values.
k_4 

    # Create a copy of the scaled DataFrame
crypto_data_scaled_df_preditcted = crypto_data_scaled_df.copy() 

    
    # Add a new column to the copy of the scaled DataFrame with the predicted clusters
crypto_data_scaled_df_preditcted["crypto_segment"] = k_4

    # Display the copy of the scaled DataFrame
crypto_data_scaled_df_preditcted.head()

![image](https://github.com/user-attachments/assets/9562d627-90ea-402f-8323-306b39a65f2d) 

    
     Create a scatter plot using hvPlot as follows:
     
        Set the x-axis as "price_change_percentage_24h" and the y-axis as "price_change_percentage_7d".
        Color the graph points with the labels found using K-means.
        Add the "coin_id" column in the hover_cols parameter to identify the cryptocurrency represented by each data point.  

crypto_data_scaled_df_preditcted.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="crypto_segment",
    hover_cols=["coinid"],
    title= "Crypto Segmentation based on K-Means Clustering (k=4)"
)

![image](https://github.com/user-attachments/assets/bdd324d6-fc60-423b-abc3-9aa976582e8b)






        

Optimize Clusters with Principal Component Analysis

    Using the original scaled DataFrame, perform a PCA and reduce the features to three principal components.
    # Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)  

    # Use the PCA model with `fit_transform` to reduce the original scaled DataFrame
    # down to three principal components.
crypto_pca = pca.fit_transform(crypto_data_scaled)

    
Retrieve the explained variance to determine how much information can be attributed to each principal component and then answer the following question in your notebook:
        What is the total explained variance of the three principal components?

    # Retrieve the explained variance to determine how much information
    # can be attributed to each principal component.
pca.explained_variance_ratio_ 


 Answer the following question: 
    What is the total explained variance of the three principal components?
    The answer is: O.89 or 89%
        
    # Create a new DataFrame with the PCA data.
crypto_pca_df = pd.DataFrame(
    crypto_pca,
    columns=["PCA1", "PCA2", "PCA3"]
)

    # Copy the crypto names from the original scaled DataFrame
crypto_pca_df['coinid'] = df_market_data.index  

    # Set the coin_id column as index
crypto_pca_df.set_index('coinid', inplace=True) 

    # Display the scaled PCA DataFrame
crypto_pca_df.head()  

![image](https://github.com/user-attachments/assets/352b8655-4a9a-487b-bfae-b46f05ad5709)




        

Find the Best Value for k Using the PCA Data

Use the elbow method on the PCA data to find the best value for k using the following steps:

    # Create a list with the number of k-values from 1 to 11
k = list(range(1, 11))
k  

    # Create an empty list to store the inertia values
inertia = [] 
    
    # Create a for loop to compute the inertia with each possible value of k
    # Inside the loop:
    # 1. Create a KMeans model using the loop counter for the n_clusters
    # 2. Fit the model to the data using `df_market_data_pca`
    # 3. Append the model.inertia_ to the inertia list

for i in k:
    k_model = KMeans(n_clusters=i, random_state=0)
    k_model.fit(crypto_pca_df)
    inertia.append(k_model.inertia_)
print(k)
print(inertia)

    
    # Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia} 

    # Create a DataFrame with the data to plot the Elbow curve
df_elbow_pca = pd.DataFrame(elbow_data) 
df_elbow_pca 


![image](https://github.com/user-attachments/assets/8632e3fb-c9e1-4778-9b19-a449f155898e)



    # Plot a line chart with all the inertia values computed with
    # the different values of k to visually identify the optimal value for k.
df_elbow_pca.hvplot.line(
    x="k",
    y="inertia",
    title="Elbow Curve PCA",
    xticks=k
)
    
    
![image](https://github.com/user-attachments/assets/aac3ffaf-d95f-44b2-884d-23a5c1645d33)

    
    
    
    
    
    
   # Answer the following question in your notebook:
        What is the best value for k when using the PCA data?
        The best value for K according to the elbow curve above is 4.  
        
        Does it differ from the best k value found using the original data?
        Yes, the best value for K does differ in this instance.
 
  
  Cluster Cryptocurrencies with K-means Using the PCA Data

## Use the following steps to cluster the cryptocurrencies for the best value for k on the PCA data:

    # Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=0) 

    # Fit the K-Means model using the PCA data
model.fit(crypto_pca_df) 

![image](https://github.com/user-attachments/assets/6d2dd256-bc5b-4b39-8486-6e2bfed78deb)  


    # Predict the clusters to group the cryptocurrencies using the scaled PCA DataFrame
k_4 = model.predict(crypto_pca_df)

    # Print the resulting array of cluster values.
k_4

    # Create a copy of the scaled PCA DataFrame
crypto_pca_predictions_df = crypto_pca_df.copy()

    # Add a new column to the copy of the PCA DataFrame with the predicted clusters
crypto_pca_predictions_df["clusters"] = k_4  

    # Display the copy of the scaled PCA DataFrame
crypto_pca_predictions_df.head()

![image](https://github.com/user-attachments/assets/d83ac7dc-b04d-42fd-9d4f-11ed3a1f8188)  

    # Create a scatter plot using hvPlot by setting
    # `x="PC1"` and `y="PC2"`.
    # Color the graph points with the labels found using K-Means and
    # add the crypto name in the `hover_cols` parameter to identify
    # the cryptocurrency represented by each data point.
crypto_pca_predictions_df.hvplot.scatter(
    x="PCA1",
    y="PCA2",
    by="clusters",
    hover_cols=["coinid"], 
    title = "Crypto Clusters" # Add the cryptocurrency name to hover information
)

![image](https://github.com/user-attachments/assets/347c995a-30ad-4162-b8b2-409e2fdf4a35)

Visualize and Compare the Results
In this section, you will vusually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

    # Composite plot to contrast the Elbow curves 
    # Create the Elbow curve plot for the original data
elbow_curve_original = df_elbow.hvplot.line(
    x="k", 
    y="inertia", 
    xlabel="Number of Clusters (k)", 
    ylabel="Inertia", 
    title="Original Elbow Curve",
    line_width=2,
    color="blue",
    legend='top_left'
)

    # Create the Elbow curve plot for the PCA data
elbow_curve_pca = df_elbow_pca.hvplot.line(
    x="k", 
    y="inertia", 
    xlabel="Number of Clusters (k)", 
    ylabel="Inertia", 
    title= "PCA elbow curve",
    line_width=2,
    color="green",
    legend='top_left'
)

    # Combine the plots into a composite plot
    # composite_plot = elbow_curve_original * elbow_curve_pca 

    # Display the composite plot
    #composite_plot  
elbow_curve_original + elbow_curve_pca  

 ![image](https://github.com/user-attachments/assets/01aba18c-0cd5-4ec1-bf2b-83ce1ca9cc5e)



    Composite plot to contrast the clusters 
orginal_cluster_plot =crypto_data_scaled_df_preditcted.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="crypto_segment",
    hover_cols=["coinid"],
    title= "Crypto Segmentation based on K-Means Clustering (k=4)"
)

pca_plot = crypto_pca_predictions_df.hvplot.scatter(
    x="PCA1",
    y="PCA2",
    by="clusters",
    hover_cols=["coinid"], 
    title = "Crypto Clusters" 
) 

orginal_cluster_plot + pca_plot 





![image](https://github.com/user-attachments/assets/ea766352-1309-4b65-9711-0c515187d326)



 
 
 ##  Answer the following question:
        What is the impact of using fewer features to cluster the data using K-Means? 
        The impact results in a reduction of noise as well as an improvement in the seperation of the 3 Visual components: segments,visualization, and the clustering performance.

