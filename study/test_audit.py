import unittest
from audit_predictions import average_precision, audit_rows, audit_manifest


class MetricChecks(unittest.TestCase):
    def test_tied_scores_have_prevalence_ap(self):
        self.assertAlmostEqual(average_precision([1, 0, 1, 0], [1, 1, 1, 1]), 0.5)
        self.assertAlmostEqual(average_precision([0, 1, 0, 1], [1, 1, 1, 1]), 0.5)

    def test_grouped_ap_matches_hand_calculation(self):
        # Top tie group has 1/2 precision and half the positives; final group has 2/3.
        self.assertAlmostEqual(average_precision([1, 0, 1], [0.8, 0.8, 0.2]),
                               0.5 * 0.5 + 0.5 * (2 / 3))

    def test_balanced_accuracy_equals_macro_recall(self):
        rows = [dict(sample_id=str(i), y_true=y, y_pred=p, score_0=1-p, score_1=p)
                for i, (y, p) in enumerate([(0, 0), (0, 1), (1, 1), (1, 1)])]
        report = audit_rows(rows, 2)
        self.assertEqual(report['accuracy'], 0.75)
        self.assertEqual(report['macro']['recall'], 0.75)

    def test_missing_classes_and_repeated_ids_fail(self):
        row = dict(sample_id='one', y_true=0, y_pred=0, score_0=1, score_1=0)
        with self.assertRaises(ValueError):
            audit_rows([row], 2)
        with self.assertRaises(ValueError):
            audit_rows([row, row], 2)

    def test_cross_split_track_fails(self):
        manifest = [dict(sample_id='a', split='train', group_id='same'),
                    dict(sample_id='b', split='test', group_id='same')]
        with self.assertRaises(ValueError):
            audit_manifest(manifest, ['b'])

    def test_absent_test_class_prediction_remains_an_error(self):
        rows=[dict(sample_id='a',y_true=0,y_pred=2,score_0=0,score_1=0,score_2=1),
              dict(sample_id='b',y_true=1,y_pred=1,score_0=0,score_1=1,score_2=0)]
        report=audit_rows(rows,3,allow_missing_test_classes=True)
        self.assertEqual(report['accuracy'],.5)
        self.assertEqual(report['macro_class_ids'],[0,1])
        self.assertEqual(report['absent_test_class_ids'],[2])
        self.assertIsNone(report['per_class'][2]['average_precision'])
        self.assertEqual(report['confusion_matrix'][0][2],1)

    def test_missing_group_metadata_is_not_verified(self):
        result = audit_manifest([dict(sample_id='a', split='test')], ['a'])
        self.assertFalse(result['checks']['group_id']['complete'])


if __name__ == '__main__':
    unittest.main()
