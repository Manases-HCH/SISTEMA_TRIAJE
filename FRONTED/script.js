const API_URL = "http://127.0.0.1:8000/api";
const $ = (id) => document.getElementById(id);
const pct = (x) => `${(x * 100).toFixed(1)}%`;

function mostrarError(msg) {
    $("resultado").hidden = true;
    $("error").textContent = msg;
    $("error").hidden = false;
}

// Estado del backend: modo demo y disponibilidad del módulo de imagen
fetch(`${API_URL}/health`).then(r => r.json()).then(h => {
    $("demoBanner").hidden = !h.demo;
    if (!h.vision_activa) {
        $("foto").disabled = true;
        $("fotoLabel").textContent = `Foto de referencia (no disponible): ${h.vision_motivo || ""}`;
    }
}).catch(() => {});

// Vista previa de la foto
$("foto").addEventListener("change", () => {
    const f = $("foto").files[0];
    $("fotoPrev").hidden = !f;
    if (f) $("fotoPrev").src = URL.createObjectURL(f);
});

$("predictionForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("error").hidden = true;

    const fd = new FormData();
    fd.append("edad", $("edad").value);
    fd.append("sexo", $("sexo").value);
    fd.append("temperatura", $("temperatura").value);
    fd.append("frecuencia_cardiaca", $("fc").value);
    fd.append("frecuencia_respiratoria", $("fr").value);
    fd.append("presion_sistolica", $("pas").value);
    fd.append("presion_diastolica", $("pad").value);
    fd.append("saturacion_o2", $("spo2").value);
    fd.append("estado_mental", $("mental").value);
    fd.append("dolor", $("dolor").value);
    const foto = $("foto").files[0];
    if (foto && !$("foto").disabled) fd.append("foto_paciente", foto);

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
        let detalle = `Riesgo estimado: ${pct(data.riesgo_ponderado)} · Datos clínicos: ${pct(data.score_clinico)}`;
        if (data.imagen_evaluada) {
            detalle += ` · Imagen: ${pct(data.score_imagen)} (pesos: clínico ${pct(data.pesos.clinico)}, imagen ${pct(data.pesos.imagen)})`;
        }
        $("scoreDetalle").textContent = detalle;
        $("aviso").textContent = data.aviso;
        $("resultado").className = `card-resultado ${data.nivel_riesgo.toLowerCase()}`;
        $("resultado").hidden = false;
    } catch (err) {
        mostrarError("No se pudo conectar con el servidor. ¿Está corriendo el backend?");
    } finally {
        btn.disabled = false; btn.textContent = "Calcular riesgo";
    }
});