def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


data_path = 'lesion_data_multiclass/test'


data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root=data_path,
    transform=data_transforms
)
train_loader = torch.utils.data.DataLoader(
    train_dataset
)


df = pd.DataFrame()

for (images, labels) in train_loader:
    images = Variable(images)
    if torch.cuda.is_available():
        images = images.cuda()

    outputs2 = model_conv(images)
    # add probabilities to DataFrame
    df = df.append(pd.Series(np.apply_along_axis(softmax, 1, outputs2.detach().numpy())[0]),ignore_index=True)
    break
    
