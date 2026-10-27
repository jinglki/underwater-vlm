def kaggle_download():
    import kagglehub
    # Download latest version
    path = kagglehub.dataset_download("tsunamiserra/icm-benchmark-20")
    return path