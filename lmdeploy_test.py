def test_lmdeploy():
    from lmdeploy import pipeline, TurbomindEngineConfig
    from lmdeploy.vl import load_image

    model = 'work_dirs/internvl_chat_v2_5/internvl2_5_8b_dynamic_res_2nd_finetune_lora_unfreeze_llm_backbone_mlp_v2'
    image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
    response = pipe(('describe this image', image))
    print(response.text)

if __name__=="__main__":
    # tansform_dot2json()
    test_lmdeploy()