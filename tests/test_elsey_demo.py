"""Smoke tests for the selected notebook demo (run after exporting models)."""
import pytest


@pytest.mark.parametrize('name', ['LR', 'RF', 'MLP'])
def test_selected_notebook_demo(monkeypatch, name):
    monkeypatch.setenv('ECS171_EXPERIMENT', 'elsey')
    from app import create_app

    app = create_app()
    client = app.test_client()
    response = client.get('/api/bootstrap')
    assert response.status_code == 200
    bootstrap = response.get_json()
    assert len(bootstrap['features']) == 16
    assert len(bootstrap['examples']) == 7
    assert '70/15/15' in bootstrap['experiment']
    result = client.post('/api/predict', json={
        'model': name, 'condition': 'compare',
        'features': bootstrap['examples'][0]['features'],
    })
    assert result.status_code == 200
    predictions = result.get_json()['predictions']
    assert [p['condition'] for p in predictions] == ['clean', 'random']
    for item in predictions:
        assert abs(sum(p['probability'] for p in item['probabilities']) - 1) < 1e-6
        assert item['class'] in bootstrap['classes']
