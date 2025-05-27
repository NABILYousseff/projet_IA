Data and information are available in the folder "material"

To understand the dataset, first read the article.pdf.

If you need some deeper information, you can also read PhDthesis.pdf (but this should not be needed)

Data are in dataset.csv. This file does not have the names of the columns. I added them in the separate file column-names.txt. When you build your dataframe, you should find a way to add such names as the header of your dataframe.

If the dataset is too big, you can just extract a subsample.

You can perform many machine learning tasks. You can predict the ITU MOS, or the startup delay, or whether there will be stalls, or how many stalls there will be, etc. It is up to you to choose one or more among these possible tasks. You can do regression or classification. Do what you want. You have just to select tasks that, put together, result in an interesting "storytelling" during your project presentation.

You can use different sets of features, and see how model accuracy changes. Indeed, if you imagine to take the role of a network operator, the network operator might not have access to all features, but only the ones that can be measured by monitoring encrypted traffic. You may compare the prediction accuracy that an operator can get, given its limited amount of features it can monitor, with the accuracy one could ideally get if one were able to measure other information, which are unfortunately not available to the operator. 
