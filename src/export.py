from rfdetr import RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRSegPreview

# model = RFDETRNano(pretrain_weights="/home/project_python/rf-detr/runs/nano/checkpoint_best_total.pth")
model=RFDETRMedium(resolution=576,num_classes=1,pretrain_weights=r'D:\aotto\budingban\model\yuqi_04-20-576m\checkpoint_best_total.pth')
model.export(output_dir=r"D:\aotto\budingban\model\yuqi_04-20-576m\export",
            # shape=(768,576),
             # simplify=True,
             batch_size=1,
             opset_version=17,)
