"""Pruebas del único método unificado de predicción y evaluación de Modelo_ANN."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.Modelo_ANN import Modelo_ANN


class Predictor:
    def __init__(self, salida):
        self.salida = np.asarray(salida)
        self.calls = 0

    def predict(self, x, verbose=0):
        self.calls += 1
        self.x = np.array(x, copy=True)
        return self.salida.copy()


def modelo(tipo, pred):
    m = Modelo_ANN(tipo=tipo)
    m.model = Predictor(pred)
    m.scaler = MinMaxScaler().fit(pd.DataFrame({'a': [0., 10.], 'b': [0., 20.]}))
    m.x_val = np.zeros((len(pred), 2))
    return m


class PrediccionUnificada(unittest.TestCase):
    def setUp(self):
        self.silencio = contextlib.redirect_stdout(io.StringIO())
        self.silencio.__enter__()

    def tearDown(self):
        plt.close('all')
        self.silencio.__exit__(None, None, None)

    def test_binaria_validacion_umbral_original(self):
        m = modelo('binaria', [[.8], [.81], [.2]])
        fechas = pd.to_datetime(['2020-01-01'] * 3)
        m.y_val = pd.Series([0, 1, 0], name='accion', index=fechas)
        self.assertIs(m.make_prediction(eval=True, graficar=False), m)
        self.assertEqual(m.predictions.accion.tolist(), [0, 1, 0])
        self.assertEqual(m.resultado['metricas'].iloc[0].accuracy, 1)
        self.assertTrue(m.predictions.index.equals(fechas))
        np.testing.assert_allclose(m.probabilidades.sum(axis=1), 1)
        np.testing.assert_array_equal(m.model.x, m.x_val)
        self.assertEqual(m.model.calls, 1)
        self.assertFalse(plt.get_fignums())

    def test_dataframe_externo_nombres_orden_sin_mutacion(self):
        m = modelo('multiclase', [[.9, .05, .05], [.7, .2, .1], [.1, .8, .1]])
        df = pd.DataFrame({'b': [2., 4., 6.], 'accion': ['a', 'b', 'b'], 'a': [1., 2., 3.], 'escenario': [1, 2, 3]},
                          index=pd.to_datetime(['2020-01-01'] * 3))
        original = df.copy(deep=True)
        m.data = original.copy(deep=True)
        m.make_prediction(data=df, target='accion', new_data=True, categorias=['a', 'b', 'c'], eval=True, graficar=False)
        reporte = m.resultado['reporte_clasificacion']
        np.testing.assert_allclose(reporte.precision, [.5, 1, 0])
        np.testing.assert_allclose(reporte.recall, [1, .5, 0])
        self.assertEqual(reporte.iloc[2]['support'], 0)
        self.assertEqual(m.resultado['matriz_confusion'].shape, (3, 3))
        np.testing.assert_allclose(m.model.x, [[.1, .1], [.2, .2], [.3, .3]])
        self.assertTrue(m.predictions.index.equals(df.index))
        pd.testing.assert_frame_equal(df, original)
        pd.testing.assert_frame_equal(m.data, original)

    def test_multiclase_indices_y_prediccion_sin_etiquetas(self):
        m = modelo('multiclase', [[.1, .2, .7], [.6, .2, .2]])
        m.y_val = np.array([2, 0])
        m.make_prediction(eval=True, graficar=False)
        self.assertEqual(m.resultado['metricas'].iloc[0].accuracy, 1)
        m.make_prediction(data=np.ones((2, 2)), escalado=True)
        np.testing.assert_array_equal(m.model.x, np.ones((2, 2)))
        self.assertNotIn('metricas', m.resultado)
        self.assertEqual(m.predictions.iloc[:, 0].tolist(), [2, 0])

    def test_regresion_varios_objetivos_desescalados(self):
        reales = np.array([[10., 20.], [20., 40.], [30., 60.]])
        pred = reales + [1., 3.]
        scaler = MinMaxScaler().fit(reales)
        m = modelo('regresion', scaler.transform(pred))
        m.y_val = scaler.transform(reales)
        m.make_prediction(target=['casos1', 'casos2'], eval=True, y_scaler=scaler, graficar=False)
        metricas = m.resultado['metricas']
        self.assertAlmostEqual(metricas.iloc[0].MAE, 2)
        self.assertAlmostEqual(metricas.iloc[0].RMSE, np.sqrt(5))
        np.testing.assert_allclose(m.predictions, pred)
        np.testing.assert_allclose(m.resultado['reales'], reales)
        self.assertIsNone(m.probabilidades)
        self.assertEqual(m.model.calls, 1)

    def test_regresion_simple_y_graficos(self):
        m = modelo('regresion', [-2.5, 3.75])
        m.y_val = [-2.5, 3.75]
        with patch('matplotlib.pyplot.show') as mostrar:
            m.make_prediction(eval=True)
            mostrar.assert_called_once()
        self.assertEqual(m.resultado['metricas'].iloc[0].RMSE, 0)
        np.testing.assert_allclose(m.predictions.iloc[:, 0], [-2.5, 3.75])
        m = modelo('binaria', [[.1], [.9]])
        m.y_val = [0, 1]
        with patch('matplotlib.pyplot.show') as mostrar:
            m.make_prediction(eval=True)
            mostrar.assert_called_once()

    def test_particion_temporal_escenarios_y_escalado(self):
        fechas = pd.date_range('2020-01-05', periods=10, freq='W')
        df = pd.DataFrame({'a': np.tile(np.arange(10), 5),
                           'recomendacion': np.tile([0, 1] * 5, 5)},
                          index=np.tile(fechas, 5))
        original = df.copy(deep=True)
        m = Modelo_ANN('multiclase')
        m.data = df
        self.assertIs(m.set_scaled_partitions(['a'], ['recomendacion'], stratify=False), m)
        self.assertEqual((len(m.x_train), len(m.x_val)), (40, 10))
        self.assertLess(m.y_train.index.max(), m.y_val.index.min())
        self.assertTrue(m.y_train.index.intersection(m.y_val.index).empty)
        self.assertEqual(m.scaler.data_max_[0], 7)
        self.assertGreater(m.x_val.max(), 1)
        pd.testing.assert_frame_equal(df, original)
        with self.assertRaises(ValueError):
            m.set_scaled_partitions(['a'], ['recomendacion'], stratify=True)

    def test_start_training_pesos_sin_atributos_de_archivos(self):
        from unittest.mock import Mock
        m = Modelo_ANN('multiclase')
        m.x_train = np.zeros((4, 2))
        m.x_val = np.zeros((2, 2))
        m.y_train = pd.DataFrame({'clase': [0, 0, 0, 1]})
        m.y_val = pd.DataFrame({'clase': [0, 1]})
        m.model = Mock()
        self.assertIs(m.start_training(epochs=1), m)
        pesos = m.model.fit.call_args.kwargs['class_weight']
        self.assertAlmostEqual(pesos[0], 2/3)
        self.assertAlmostEqual(pesos[1], 2)
        m.start_training(epochs=1, w_balance=False)
        self.assertIsNone(m.model.fit.call_args.kwargs['class_weight'])
        m.start_training(epochs=1, w_dict={0: 3., 1: 4.})
        self.assertEqual(m.model.fit.call_args.kwargs['class_weight'], {0: 3., 1: 4.})

    def test_errores_de_datos_y_categorias(self):
        m = modelo('multiclase', [[.1, .9], [.9, .1]])
        for kwargs in ({'eval': True}, {'new_data': True},
                       {'data': np.empty((0, 2))}, {'data': [[np.nan, 0], [0, 0]]},
                       {'data': pd.DataFrame([[1, 2]], columns=['a', 'a'])},
                       {'data': pd.DataFrame({'a': [1]})},
                       {'data': [[1, 2]], 'escalado': True},
                       {'categorias': ['a', 'a']}, {'categorias': ['a']},
                       {'categorias': 'ab'}, {'eval': True, 'y': ['a', 'b']},
                       {'eval': True, 'y': [0]}, {'y_scaler': MinMaxScaler()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                m.make_prediction(**kwargs)
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[1, 2])
        with self.assertRaises(ValueError):
            m.make_prediction(data=df, y=pd.Series([0, 1], index=[2, 1]), eval=True)
        for salidas in ([[np.nan], [.5]], [[-.1], [.5]], [[1.1], [.5]], [[.5, .5], [.5, .5]]):
            m = modelo('binaria', salidas)
            with self.assertRaises(ValueError): m.make_prediction()
        m = modelo('multiclase', [[.1, .1], [.2, .2]])
        with self.assertRaises(ValueError): m.make_prediction()
        m = modelo('regresion', [[1], [2]])
        with self.assertRaises(ValueError): m.make_prediction(eval=True, y=[1, np.inf])
        with self.assertRaises(ValueError): m.make_prediction(y_scaler=MinMaxScaler())


if __name__ == '__main__':
    unittest.main()
