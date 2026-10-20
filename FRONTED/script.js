const API_URL = "http://127.0.0.1:8000/api";
const $ = (id) => document.getElementById(id);

function mostrarError(msg) {
    $("resultado").hidden = true;
    $("error").textContent = msg;
    $("error").hidden = false;
}

// Aviso si el backend está en modo demo
fetch(`${API_URL}/health`).then(r => r.json()).then(h => { $("demoBanner").hidden = !h.demo; }).catch(() => {});

$("predictionForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("error").hidden = true;

    const fd = new FormData();
    fd.append("edad", $("edad").value);
    fd.append("sexo", $("sexo").value);
    fd.append("temperatura", $("temperatura").value);
    fd.append("frecuencia_cardiaca", $("fc").value);
    fd.append("presion_sistolica", $("pas").value);
    fd.append("presion_diastolica", $("pad").value);
    fd.append("saturacion_o2", $("spo2").value);
    fd.append("sintoma_severidad", $("sintoma").value);
    const foto = $("foto").files[0];
    if (foto) fd.append("foto_paciente", foto);

    const btn = $("btnEnviar");
    btn.disabled = true; btn.textContent = "Calculando...";
    try {
        const resp = await fetch(`${API_URL}/predict`, { method: "POST", body: fd });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
            const detalle = Array.isArray(data.detail)
                ? data.detail.map(d => `${d.loc.slice(-1)[0]}: ${d.msg}`).join(" | ")
                : (data.detail || `Error ${resp.status}`);
            return mostrarError(detalle);
        }
        $("nivelRiesgo").textContent = `Nivel de riesgo: ${data.nivel_riesgo}`;
        $("recomendacion").textContent = data.recomendacion;
        $("alertaVitales").hidden = !data.alerta_signos_vitales;
        $("scoreDetalle").textContent =
            `Índice estimado: ${(data.riesgo_ponderado * 100).toFixed(1)}%` +
            (foto && !data.imagen_evaluada ? " · (la foto no se usó: modelo de imagen no disponible)" : "");
        $("aviso").textContent = data.aviso;
        $("resultado").className = `card-resultado ${data.nivel_riesgo.toLowerCase()}`;
        $("resultado").hidden = false;
    } catch (err) {
        mostrarError("No se pudo conectar con el servidor. ¿Está corriendo el backend?");
    } finally {
        btn.disabled = false; btn.textContent = "Calcular riesgo";
    }
});