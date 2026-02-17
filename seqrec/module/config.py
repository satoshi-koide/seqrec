from transformers import PretrainedConfig

class SeqRecBaseConfig(PretrainedConfig):
    """
    Sequential Recommendation モデルの基底設定クラス
    全てのモデルで共通するパラメータ（語彙サイズ、次元数、付加情報など）を管理する
    """
    model_type = "seqrec"

    def __init__(
        self,
        num_items=50000,
        max_len=50,
        hidden_units=64,
        dropout_rate=0.2,
        use_rating=False,
        image_feature_dim=0,
        text_feature_dim=0,
        pad_token_id=0,
        use_cache=False,  # Trainer対策
        **kwargs
    ):
        """
        Args:
            num_items (int): アイテムの総数（パディング含む）
            max_len (int): 系列の最大長
            hidden_units (int): 埋め込みベクトルの次元数
            dropout_rate (float): ドロップアウト率
            use_rating (bool): Rating情報を使用するか
            image_feature_dim (int): 画像特徴量の次元（0なら使用しない）
            text_feature_dim (int): テキスト特徴量の次元（0なら使用しない）
            pad_token_id (int): パディングID (デフォルト: 0)
            use_cache (bool): 推論時にKVキャッシュを使うか (SASRecでは通常使わない)
        """
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.use_rating = use_rating
        self.image_feature_dim = image_feature_dim
        self.text_feature_dim = text_feature_dim
        self.use_cache = use_cache

        # PretrainedConfig に標準パラメータ（pad_token_idなど）を渡す
        super().__init__(pad_token_id=pad_token_id, **kwargs)
