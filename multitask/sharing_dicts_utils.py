sharing_dict_soft = {
    'Attention': False,
    'Decoder': [False, False],
    'EncoderBW': [False, False],
    'EncoderFW': [False, False],
    'Pointer': False,
    'Projection': False,
    'WordEmb': False}

 
Attention_Params = [
    'Newsela_Attention/memory_kernel/kernel:0',
    'Newsela_Attention/input_kernel/kernel:0',
    'Newsela_Attention/query_kernel/kernel:0',
    # 'Newsela_Attention/coverage_kernel/kernel:0',
    'Newsela_Attention/attention_v:0',
    'Newsela_Attention/output_kernel/kernel:0',

    'WikiSmall_Attention/memory_kernel/kernel:0',
    'WikiSmall_Attention/input_kernel/kernel:0',
    'WikiSmall_Attention/query_kernel/kernel:0',
    # 'WikiSmall_Attention/coverage_kernel/kernel:0',
    'WikiSmall_Attention/attention_v:0',
    'WikiSmall_Attention/output_kernel/kernel:0',

    'WikiLarge_Attention/memory_kernel/kernel:0',
    'WikiLarge_Attention/input_kernel/kernel:0',
    'WikiLarge_Attention/query_kernel/kernel:0',
    # 'WikiLarge_Attention/coverage_kernel/kernel:0',
    'WikiLarge_Attention/attention_v:0',
    'WikiLarge_Attention/output_kernel/kernel:0',

    'SNLI_Attention/memory_kernel/kernel:0',
    'SNLI_Attention/input_kernel/kernel:0',
    'SNLI_Attention/query_kernel/kernel:0',
    # 'SNLI_Attention/coverage_kernel/kernel:0',
    'SNLI_Attention/attention_v:0',
    'SNLI_Attention/output_kernel/kernel:0',

    'PP_Attention/memory_kernel/kernel:0',
    'PP_Attention/input_kernel/kernel:0',
    'PP_Attention/query_kernel/kernel:0',
    # 'PP_Attention/coverage_kernel/kernel:0',
    'PP_Attention/attention_v:0',
    'PP_Attention/output_kernel/kernel:0']


Encoder_LowerLayer_Params = [
    'Newsela_EncoderBW_0/lstm_cell/kernel:0',
    'Newsela_EncoderFW_0/lstm_cell/kernel:0',
    'WikiSmall_EncoderBW_0/lstm_cell/kernel:0',
    'WikiSmall_EncoderFW_0/lstm_cell/kernel:0',
    'WikiLarge_EncoderBW_0/lstm_cell/kernel:0',
    'WikiLarge_EncoderFW_0/lstm_cell/kernel:0',
    'PP_EncoderBW_0/lstm_cell/kernel:0',
    'PP_EncoderFW_0/lstm_cell/kernel:0',
    'SNLI_EncoderBW_0/lstm_cell/kernel:0',
    'SNLI_EncoderFW_0/lstm_cell/kernel:0']


Encoder_HigherLayer_Params = [
    'Newsela_EncoderBW_1/lstm_cell/kernel:0',
    'Newsela_EncoderFW_1/lstm_cell/kernel:0',
    'WikiSmall_EncoderBW_1/lstm_cell/kernel:0',
    'WikiSmall_EncoderFW_1/lstm_cell/kernel:0',
    'WikiLarge_EncoderBW_1/lstm_cell/kernel:0',
    'WikiLarge_EncoderFW_1/lstm_cell/kernel:0',
    'PP_EncoderBW_1/lstm_cell/kernel:0',
    'PP_EncoderFW_1/lstm_cell/kernel:0',
    'SNLI_EncoderBW_1/lstm_cell/kernel:0',
    'SNLI_EncoderFW_1/lstm_cell/kernel:0']

Decoder_HigherLayer_Params = [
    'Newsela_Decoder_0/lstm_cell/kernel:0',
    'WikiSmall_Decoder_0/lstm_cell/kernel:0',
    'WikiLarge_Decoder_0/lstm_cell/kernel:0',
    'PP_Decoder_0/lstm_cell/kernel:0',
    'SNLI_Decoder_0/lstm_cell/kernel:0']

Decoder_LowerLayer_Params = [
    'Newsela_Decoder_1/lstm_cell/kernel:0',
    'WikiSmall_Decoder_1/lstm_cell/kernel:0',
    'WikiLarge_Decoder_1/lstm_cell/kernel:0',
    'PP_Decoder_1/lstm_cell/kernel:0',
    'SNLI_Decoder_1/lstm_cell/kernel:0']
 

Layered_Shared_Params = (
    Attention_Params +
    Encoder_HigherLayer_Params +
    Decoder_HigherLayer_Params)

E1D2_Shared_Params = (
    Encoder_LowerLayer_Params +
    Decoder_LowerLayer_Params)
