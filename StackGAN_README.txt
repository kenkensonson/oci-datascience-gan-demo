# Oracle Big Data Jam Session 「StackGANの概要とデモンストレーション」
## 1. ファイルの説明
- 20210531_OraJam_StackGAN.pdf : 説明に利用したスライド
- 01_StackGAN_test1.ipynb, 01_StackGAN_test2.ipynb : デモに利用したnotebookファイル

## 2. StackGAN (Python3系)
 - StackGANのフォルダを作成後、その配下にgit cloneまたはdownloadを行う
 - model, dataのダウンロードなど詳細については、https://github.com/anthonyftwang/StackGAN-Pytorch-Python3 を参照
 - notebook「01_StackGAN_test1.ipynb」,「02_StackGAN_test1.ipynb」も同じくStackGAN/配下に置く
 - StackGAN/StackGAN-Pytorch-Python3/code/trainer.pyのsample関数を以下に置き換える</br>
   参考サイト: http://cedro3.com/ai/pytorch-stackgan/
 - 変数countはnotebook毎に変える（コメント記載）。

```
def sample(self, datapath, stage=1):
    if stage == 1:
        netG, _ = self.load_network_stageI()
    else:
        netG, _ = self.load_network_stageII()
    netG.eval()

    # Load text embeddings generated from the encoder
    t_file = torchfile.load(datapath)
    captions_list = t_file.raw_txt
    embeddings = np.concatenate(t_file.fea_txt, axis=0)
    num_embeddings = len(captions_list)
    print('Successfully load sentences from: ', datapath)
    print('Total number of sentences:', num_embeddings)
    print('num_embeddings:', num_embeddings, embeddings.shape)
    # path to save generated samples
    save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
    mkdir_p(save_dir)

    batch_size = 1
    nz = cfg.Z_DIM
    noise = Variable(torch.FloatTensor(batch_size, nz))
    if cfg.CUDA:
        noise = noise.cuda()
        
    count = [816,1334,1370,2886] # 01_StackGAN_test1.ipynbの場合
    #count = [3,11,44,249] # 01_StackGAN_test2.ipynbの場合
    for k in range(len(count)):
        embeddings_batch = embeddings[count[k]:count[k]+1]  
        # captions_batch = captions_list[count:iend]
        txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
        if cfg.CUDA:
            txt_embedding = txt_embedding.cuda()

        #######################################################
        # (2) Generate fake images
        ######################################################
        #print(txt_embedding)
        print(count[k],captions_list[count[k]])
        for i in range(10): # batch_size to 10
            noise.data.normal_(0, 1)
            inputs = (txt_embedding, noise)
            _, fake_imgs, mu, logvar = \
            nn.parallel.data_parallel(netG, inputs, self.gpus)
            save_name = '%s/%d.png' % (save_dir, count[k]*10 + i)  
            im = fake_imgs[0].data.cpu().numpy()  
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            # print('im', im.shape)
            im = np.transpose(im, (1, 2, 0))
            # print('im', im.shape)
            im = Image.fromarray(im)
            im.save(save_name) 