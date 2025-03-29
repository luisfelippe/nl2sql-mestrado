import argparse
import os
import json
import sqlite3

from src.env import LEVELS, PARTIAL_TYPES
from src.evaluation.base_evaluator_utils import BaseEvaluatorUtils
from src.evaluation.spider_lib.process_sql import get_schema, Schema, get_sql

class Evaluator(BaseEvaluatorUtils):
    """
    Um simples avaliador de consultas SQL,
    baseado na classe Evaluator do benchmark Spider.
    """

    def __init__(self, db_dir, kmaps, etype):
        self.db_dir = db_dir
        self.kmaps = kmaps
        self.etype = etype

        self.db_paths = {}
        self.schemas = {}

        for db_name in self.kmaps.keys():
            db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")

            self.db_paths[db_name] = db_path

            self.schemas[db_name] = Schema(get_schema(db_path))

        self.scores = {
            level: {
                "count": 0,
                "partial": {
                    type_: {
                        "acc": 0.0,
                        "rec": 0.0,
                        "f1": 0.0,
                        "acc_count": 0,
                        "rec_count": 0,
                    }
                    for type_ in PARTIAL_TYPES
                },
                "exact": 0.0,
                "exec": 0,
            }
            for level in LEVELS
        }

    @classmethod
    def eval_sel(cls, pred, label):
        pred_sel = pred["select"][1]
        label_sel = label["select"][1]
        label_wo_agg = [unit[1] for unit in label_sel]
        pred_total = len(pred_sel)
        label_total = len(label_sel)
        cnt = 0
        cnt_wo_agg = 0

        for unit in pred_sel:
            if unit in label_sel:
                cnt += 1
                label_sel.remove(unit)

            if unit[1] in label_wo_agg:
                cnt_wo_agg += 1
                label_wo_agg.remove(unit[1])

        return label_total, pred_total, cnt, cnt_wo_agg

    @classmethod
    def eval_where(cls, pred, label):
        pred_conds = [unit for unit in pred["where"][::2]]
        label_conds = [unit for unit in label["where"][::2]]
        label_wo_agg = [unit[2] for unit in label_conds]
        pred_total = len(pred_conds)
        label_total = len(label_conds)
        cnt = 0
        cnt_wo_agg = 0

        for unit in pred_conds:
            if unit in label_conds:
                cnt += 1
                label_conds.remove(unit)

            if unit[2] in label_wo_agg:
                cnt_wo_agg += 1
                label_wo_agg.remove(unit[2])

        return label_total, pred_total, cnt, cnt_wo_agg

    @classmethod
    def eval_group(cls, pred, label):
        pred_cols = [unit[1] for unit in pred["groupBy"]]
        label_cols = [unit[1] for unit in label["groupBy"]]
        pred_total = len(pred_cols)
        label_total = len(label_cols)
        cnt = 0
        pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
        label_cols = [
            label.split(".")[1] if "." in label else label for label in label_cols
        ]

        for col in pred_cols:
            if col in label_cols:
                cnt += 1
                label_cols.remove(col)

        return label_total, pred_total, cnt

    @classmethod
    def eval_having(cls, pred, label):
        pred_total = label_total = cnt = 0

        if len(pred["groupBy"]) > 0:
            pred_total = 1

        if len(label["groupBy"]) > 0:
            label_total = 1

        pred_cols = [unit[1] for unit in pred["groupBy"]]
        label_cols = [unit[1] for unit in label["groupBy"]]

        if (
            pred_total == label_total == 1
            and pred_cols == label_cols
            and pred["having"] == label["having"]
        ):
            cnt = 1

        return label_total, pred_total, cnt

    @classmethod
    def eval_order(cls, pred, label):
        pred_total = label_total = cnt = 0

        if len(pred["orderBy"]) > 0:
            pred_total = 1

        if len(label["orderBy"]) > 0:
            label_total = 1

        if (
            len(label["orderBy"]) > 0
            and pred["orderBy"] == label["orderBy"]
            and (
                (pred["limit"] is None and label["limit"] is None)
                or (pred["limit"] is not None and label["limit"] is not None)
            )
        ):
            cnt = 1

        return label_total, pred_total, cnt

    @classmethod
    def eval_and_or(cls, pred, label):
        pred_ao = pred["where"][1::2]
        label_ao = label["where"][1::2]
        pred_ao = set(pred_ao)
        label_ao = set(label_ao)

        if pred_ao == label_ao:
            return 1, 1, 1

        return len(pred_ao), len(label_ao), 0

    @classmethod
    def eval_nested(cls, pred, label):
        label_total = 0
        pred_total = 0
        cnt = 0

        if pred is not None:
            pred_total += 1

        if label is not None:
            label_total += 1

        if pred is not None and label is not None:
            partial_scores = Evaluator.eval_partial_match(pred, label)
            cnt += Evaluator.eval_exact_match(pred, label, partial_scores)

        return label_total, pred_total, cnt

    @classmethod
    def eval_IUEN(cls, pred, label):
        lt1, pt1, cnt1 = cls.eval_nested(pred["intersect"], label["intersect"])
        lt2, pt2, cnt2 = cls.eval_nested(pred["except"], label["except"])
        lt3, pt3, cnt3 = cls.eval_nested(pred["union"], label["union"])

        label_total = lt1 + lt2 + lt3
        pred_total = pt1 + pt2 + pt3
        cnt = cnt1 + cnt2 + cnt3

        return label_total, pred_total, cnt

    @classmethod
    def eval_keywords(cls, pred, label):
        pred_keywords = BaseEvaluatorUtils.get_keywords(pred)
        label_keywords = BaseEvaluatorUtils.get_keywords(label)
        pred_total = len(pred_keywords)
        label_total = len(label_keywords)
        cnt = 0

        for k in pred_keywords:
            if k in label_keywords:
                cnt += 1
        return label_total, pred_total, cnt

    @classmethod
    def eval_hardness(cls, sql):
        count_comp1_ = cls.count_component1(sql)
        count_comp2_ = cls.count_component2(sql)
        count_others_ = cls.count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        
        if (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or (
            count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0
        ):
            return "medium"
        
        if (
            (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0)
            or (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0)
            or (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1)
        ):
            return "hard"
        
        return "extra"

    @classmethod
    def eval_exact_match(cls, pred, label, partial_scores):
        """
        Avalia se a consulta predita é exatamente igual à consulta rotulada.
        :param pred: consulta predita
        :param label: consulta rotulada
        :param partial_scores: pontuação parcial
        :return: 1 se a consulta predita é exatamente igual à consulta rotulada, 0 caso contrário
        """
        for _, score in list(partial_scores.items()):
            if score["f1"] != 1:
                return 0

        if len(label["from"]["table_units"]) > 0:
            label_tables = sorted(label["from"]["table_units"])
            pred_tables = sorted(pred["from"]["table_units"])

            return label_tables == pred_tables

        return 1

    @classmethod
    def eval_partial_match(cls, pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = cls.eval_sel(pred, label)
        acc, rec, f1 = cls.get_scores(cnt, pred_total, label_total)
        res["select"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }
        acc, rec, f1 = cls.get_scores(cnt_wo_agg, pred_total, label_total)
        res["select(no AGG)"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt, cnt_wo_agg = cls.eval_where(pred, label)
        acc, rec, f1 = cls.get_scores(cnt, pred_total, label_total)
        res["where"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }
        acc, rec, f1 = cls.get_scores(cnt_wo_agg, pred_total, label_total)
        res["where(no OP)"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = cls.eval_group(pred, label)
        acc, rec, f1 = cls.get_scores(cnt, pred_total, label_total)
        res["group(no Having)"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = cls.eval_having(pred, label)
        acc, rec, f1 = cls.get_scores(cnt, pred_total, label_total)
        res["group"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = cls.eval_order(pred, label)
        acc, rec, f1 = cls.get_scores(cnt, pred_total, label_total)
        res["order"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = cls.eval_and_or(pred, label)
        acc, rec, f1 = cls.get_scores(cnt, pred_total, label_total)
        res["and/or"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = cls.eval_IUEN(pred, label)
        acc, rec, f1 = cls.get_scores(cnt, pred_total, label_total)
        res["IUEN"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = cls.eval_keywords(pred, label)
        acc, rec, f1 = cls.get_scores(cnt, pred_total, label_total)
        res["keywords"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        return res

    @classmethod
    def eval_exec_match(cls, db, p_str, g_str, pred, gold):
        """
        return 1 if the values between prediction and gold are matching
        in the corresponding index. Currently not support multiple col_unit(pairs).
        """

        def res_map(res, val_units):
            rmap = {}
            for idx, val_unit in enumerate(val_units):
                key = (
                    tuple(val_unit[1])
                    if not val_unit[2]
                    else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
                )
                rmap[key] = [r[idx] for r in res]

            return rmap

        conn = sqlite3.connect(db)
        conn.text_factory = lambda x: str(x, "latin1")
        cursor = conn.cursor()

        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
        except:
            return False

        cursor.execute(g_str)
        q_res = cursor.fetchall()

        p_val_units = [unit[1] for unit in pred["select"][1]]
        q_val_units = [unit[1] for unit in gold["select"][1]]

        return res_map(p_res, p_val_units) == res_map(q_res, q_val_units)

    def evaluate_one(self, db_name, gold, predicted):
        """
        Avalia uma única consulta SQL.
        :param db_name: nome do banco de dados
        :param gold: consulta SQL rotulada
        :param predicted: consulta SQL predita
        :return: um dicionário com as consultas rotulada e predita, a dificuldade da consulta, a pontuação exata e a pontuação parcial
        """

        schema = self.schemas[db_name]

        g_sql = get_sql(schema, gold)

        hardness = self.eval_hardness(g_sql)

        self.scores[hardness]["count"] += 1
        self.scores["all"]["count"] += 1

        parse_error = False

        try:
            p_sql = get_sql(schema, predicted)
        except:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            p_sql = {
                "except": None,
                "from": {"conds": [], "table_units": []},
                "groupBy": [],
                "having": [],
                "intersect": None,
                "limit": None,
                "orderBy": [],
                "select": [False, []],
                "union": None,
                "where": [],
            }

            # TODO fix
            parse_error = True

        # rebuild sql for value evaluation
        kmap = self.kmaps[db_name]

        g_valid_col_units = self.build_valid_col_units(
            g_sql["from"]["table_units"], schema
        )
        g_sql = self.rebuild_sql_val(g_sql)
        g_sql = self.rebuild_sql_col(g_valid_col_units, g_sql, kmap)

        p_valid_col_units = self.build_valid_col_units(
            p_sql["from"]["table_units"], schema
        )
        p_sql = self.rebuild_sql_val(p_sql)
        p_sql = self.rebuild_sql_col(p_valid_col_units, p_sql, kmap)


        exec_score = 0
        if self.etype in ["all", "exec"]:
            exec_score = self.eval_exec_match(
                self.db_paths[db_name], predicted, gold, p_sql, g_sql
            )

            self.scores[hardness]["exec"] += exec_score
            self.scores["all"]["exec"] += exec_score

        exact_score = 0
        partial_scores = None

        if self.etype in ["all", "match"]:
            partial_scores = self.eval_partial_match(p_sql, g_sql)
            exact_score = self.eval_exact_match(p_sql, g_sql, partial_scores)

            self.scores[hardness]["exact"] += exact_score
            self.scores["all"]["exact"] += exact_score

            for type_ in PARTIAL_TYPES:
                if partial_scores[type_]["pred_total"] > 0:
                    self.scores[hardness]["partial"][type_]["acc"] += partial_scores[
                        type_
                    ]["acc"]
                    self.scores[hardness]["partial"][type_]["acc_count"] += 1

                if partial_scores[type_]["label_total"] > 0:
                    self.scores[hardness]["partial"][type_]["rec"] += partial_scores[
                        type_
                    ]["rec"]
                    self.scores[hardness]["partial"][type_]["rec_count"] += 1
                self.scores[hardness]["partial"][type_]["f1"] += partial_scores[type_][
                    "f1"
                ]

                if partial_scores[type_]["pred_total"] > 0:
                    self.scores["all"]["partial"][type_]["acc"] += partial_scores[
                        type_
                    ]["acc"]
                    self.scores["all"]["partial"][type_]["acc_count"] += 1

                if partial_scores[type_]["label_total"] > 0:
                    self.scores["all"]["partial"][type_]["rec"] += partial_scores[
                        type_
                    ]["rec"]
                    self.scores["all"]["partial"][type_]["rec_count"] += 1
                self.scores["all"]["partial"][type_]["f1"] += partial_scores[type_][
                    "f1"
                ]
        
        return {
            "predicted": predicted,
            "gold": gold,
            "predicted_parse_error": parse_error,
            "hardness": hardness,
            "exec": exec_score,
            "exact": exact_score,
            "partial": partial_scores,
        }

    def finalize(self):
        scores = self.scores

        for level in LEVELS:
            if scores[level]["count"] == 0:
                continue

            if self.etype in ["all", "exec"]:
                scores[level]["exec"] /= scores[level]["count"]

            if self.etype in ["all", "match"]:
                scores[level]["exact"] /= scores[level]["count"]

                for type_ in PARTIAL_TYPES:
                    if scores[level]["partial"][type_]["acc_count"] == 0:
                        scores[level]["partial"][type_]["acc"] = 0
                    else:
                        scores[level]["partial"][type_]["acc"] = (
                            scores[level]["partial"][type_]["acc"]
                            / scores[level]["partial"][type_]["acc_count"]
                            * 1.0
                        )
                    if scores[level]["partial"][type_]["rec_count"] == 0:
                        scores[level]["partial"][type_]["rec"] = 0
                    else:
                        scores[level]["partial"][type_]["rec"] = (
                            scores[level]["partial"][type_]["rec"]
                            / scores[level]["partial"][type_]["rec_count"]
                            * 1.0
                        )
                    if (
                        scores[level]["partial"][type_]["acc"] == 0
                        and scores[level]["partial"][type_]["rec"] == 0
                    ):
                        scores[level]["partial"][type_]["f1"] = 1
                    else:
                        scores[level]["partial"][type_]["f1"] = (
                            2.0
                            * scores[level]["partial"][type_]["acc"]
                            * scores[level]["partial"][type_]["rec"]
                            / (
                                scores[level]["partial"][type_]["rec"]
                                + scores[level]["partial"][type_]["acc"]
                            )
                        )

    def print_scores(self):
        ALT_LEVELS = LEVELS[:-1]+["hits/all"]

        print("#" * 114)
        print("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *[level.upper() for level in LEVELS]))

        counts = [self.scores[level]["count"] for level in LEVELS]

        print("{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}".format("count", *counts))        

        if self.etype in ["all", "exec"]:
            # print(
            #     "\n=====================   EXECUTION ACCURACY     ====================="
            # )
            header_text = " EXECUTION ACCURACY "
            print(header_text.center(114, '#'))

            print("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *[level.upper() for level in ALT_LEVELS]))

            this_scores = [self.scores[level]["exec"] for level in LEVELS]

            print(
                "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                    "execution", *this_scores
                )
            )

        if self.etype in ["all", "match"]:
            # print(
            #     "\n====================== EXACT MATCHING ACCURACY ====================="
            # )
            header_text = " EXACT MATCHING ACCURACY "
            print(header_text.center(114, '#'))

            print("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *[level.upper() for level in ALT_LEVELS]))

            exact_scores = [self.scores[level]["exact"] for level in LEVELS]

            print(
                "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                    "exact match", *exact_scores
                )
            )
            # print(
            #     "\n---------------------PARTIAL MATCHING ACCURACY----------------------"
            # )
            header_text = " PARTIAL MATCHING ACCURACY "
            print(header_text.center(114, '='))

            for type_ in PARTIAL_TYPES:
                this_scores = [
                    self.scores[level]["partial"][type_]["acc"] for level in LEVELS
                ]

                print(
                    "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                        type_, *this_scores
                    )
                )

            # print(
            #     "---------------------- PARTIAL MATCHING RECALL ----------------------"
            # )
            header_text = " PARTIAL MATCHING RECALL "
            print(header_text.center(114, '='))

            for type_ in PARTIAL_TYPES:
                this_scores = [
                    self.scores[level]["partial"][type_]["rec"] for level in LEVELS
                ]

                print(
                    "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                        type_, *this_scores
                    )
                )

            # print(
            #     "---------------------- PARTIAL MATCHING F1 --------------------------"
            # )
            header_text = " PARTIAL MATCHING F1 "
            print(header_text.center(114, '='))

            for type_ in PARTIAL_TYPES:
                this_scores = [
                    self.scores[level]["partial"][type_]["f1"] for level in LEVELS
                ]

                print(
                    "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                        type_, *this_scores
                    )
                )
        print("#" * 114)

def evaluate(predict, db_dir, etype, table, output=None):
    assert etype in ["all", "exec", "match"], "Unknown evaluation method"

    kmaps = BaseEvaluatorUtils.build_foreign_key_map_from_json(table)

    with open(predict, encoding="utf8", mode="r") as f:
        predictions = json.load(f)

    evaluator = Evaluator(db_dir, kmaps, etype)

    results = []

    for p in predictions:
        predicted = p["sql_predicted"]
        gold = p["sql_expected"]
        db_name = p["db_id"]
        results.append(evaluator.evaluate_one(db_name, gold, predicted))

    evaluator.finalize()
    evaluator.print_scores()

    results = {
        "per_item": results,
        "total_scores": evaluator.scores,
    }

    if output:
        with open(output, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", dest="pred", type=str)
    parser.add_argument("--db", dest="db", type=str)
    parser.add_argument("--table", dest="table", type=str)
    parser.add_argument("--etype", dest="etype", type=str)
    parser.add_argument("--output")
    args = parser.parse_args()

    pred = args.pred
    db_dir = args.db
    table = args.table
    etype = args.etype  

    evaluate(pred, db_dir, etype, table, args.output)
