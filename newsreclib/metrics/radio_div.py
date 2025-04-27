from newsreclib.metrics.RADio.metric import DiversityMetric
from collections import defaultdict
import ast


def compute_calibration(df_news, df_beh, recommendations):
    # Prepare mappings
    #df_beh["uid"] = df_beh["uid"].astype(str).radd("U")
    nid_to_category = df_news.set_index('nid')['category'].to_dict()
    uid_to_history = dict(zip(df_beh['uid'], df_beh['history']))

    # Prepare user aggregates
    user_agg = {}
    for c, (uid, recs) in enumerate(recommendations.items()):
        # Filter and map predicted nids to categories
        pred_category = [nid_to_category[nid] for nid, score in recs.items() if score > 0 and nid in nid_to_category]

        # Get history and map to categories
        histories = ast.literal_eval(uid_to_history.get(uid, '[]'))
        history_category = [nid_to_category[nid] for nid in histories if nid in nid_to_category]

        if history_category and pred_category:
            user_agg[uid] = {
                "hist_cat": list(set(history_category)),
                "pred_cat": list(set(pred_category)),
            }

    # Compute calibration score
    Calibration = DiversityMetric(
        feature_type='cat',
        rank_aware_recommendation=True,
        rank_aware_context=True,
        divergence='JSD',
        context='dynamic'
    )

    cal_scores = [Calibration.compute(data["hist_cat"], data["pred_cat"]) for data in user_agg.values()]
    return sum(cal_scores) / len(cal_scores) if cal_scores else 0



def compute_representation(df_news, df_beh, recommendations):
    # Preprocess
    #df_beh["uid"] = df_beh["uid"].astype(str).radd("U")
    nid_to_frame = df_news.set_index("nid")["frame_class"].to_dict()
    uid_to_history = dict(zip(df_beh["uid"], df_beh["history"]))

    user_agg = {}
    for c, (uid, recs) in enumerate(recommendations.items()):
        # Get predicted frame classes
        pred_category = [
            nid_to_frame[nid]
            for nid, score in recs.items()
            if score > 0 and nid in nid_to_frame
        ]

        # Get history frame classes
        histories = ast.literal_eval(uid_to_history.get(uid, "[]"))
        history_category = [
            nid_to_frame[nid]
            for nid in histories
            if nid in nid_to_frame
        ]

        if history_category and pred_category:
            user_agg[uid] = {
                "hist_cat": list(set(history_category)),
                "pred_cat": list(set(pred_category)),
            }

    # Compute representation score
    Representation = DiversityMetric(
        feature_type='cat', 
        rank_aware_recommendation=True,
        rank_aware_context=True,
        divergence='JSD',
        context='dynamic'
    )

    rep_scores = [
        Representation.compute(data["hist_cat"], data["pred_cat"])
        for data in user_agg.values()
    ]
    return sum(rep_scores) / len(rep_scores) if rep_scores else 0


def compute_activation(df_news, df_beh, recommendations):
    # Preprocessing
    #df_beh["uid"] = df_beh["uid"].astype(str).radd("U")
    nid_to_sentiment = df_news.set_index("nid")["sentiment_score"].abs().to_dict()
    uid_to_history = dict(zip(df_beh["uid"], df_beh["history"]))

    user_agg = {}

    for c, (uid, recs) in enumerate(recommendations.items()):
        pred_category = [
            nid_to_sentiment[nid]
            for nid, score in recs.items()
            if score > 0 and nid in nid_to_sentiment
        ]

        candidate_category = [
            nid_to_sentiment[nid]
            for nid, score in recs.items()
            if nid in nid_to_sentiment
        ]

        if candidate_category and pred_category:
            user_agg[uid] = {
                "cand_act": candidate_category,
                "pred_act": pred_category,
            }

    Activation = DiversityMetric(
        feature_type='cont',
        rank_aware_recommendation=True,
        rank_aware_context=False,
        divergence='JSD',
        bins=10,
        context='static'
    )

    act_scores = [
        Activation.compute(data["pred_act"], data["cand_act"])
        for data in user_agg.values()
    ]

    return sum(act_scores) / len(act_scores) if act_scores else 0
