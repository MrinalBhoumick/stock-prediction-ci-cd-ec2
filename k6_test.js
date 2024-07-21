import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
    vus: 5,
    duration: '30s',
};

export default function () {
    let response = http.post('http://127.0.0.1:8000/predict', JSON.stringify({ ticker: 'TCS.NS' }), {
        headers: { 'Content-Type': 'application/json' },
    });

    check(response, {
        'is status 200': (r) => r.status === 200,
    });

    sleep(1);
}
