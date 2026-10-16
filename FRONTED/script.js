document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append("edad", document.getElementById('edad').value);
    formData.append("sexo", document.getElementById('sexo').value);
    formData.append("temperatura", document.getElementById('temperatura').value);
    formData.append("frecuencia_cardiaca", document.getElementById('fc').value);
    formData.append("presion_sistolica", document.getElementById('pas').value);
    formData.append("presion_diastolica", document.getElementById('pad').value);
    formData.append("sintoma_severidad", document.getElementById('sintoma').value);

    const foto = document.getElementById('foto').files[0];
    if (foto) {
        formData.append("foto_paciente", foto);
    }

    try {
        const response = await fetch("http://127.0.0.1:8000/api/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        const resBox = document.getElementById('resultado');
        resBox.style.display = "block";
        
        document.getElementById('nivelRiesgo').textContent = `Nivel de riesgo: ${data.nivel_riesgo}`;
        document.getElementById('recomendacion').textContent = data.recomendacion;
        document.getElementById('scoreDetalle').textContent = `Índice estimado: ${(data.riesgo_ponderado * 100).toFixed(1)}%`;
    } catch (error) {
        console.error("Error al procesar la predicción:", error);
    }
});