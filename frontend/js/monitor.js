/** System monitoring — polls backend for hardware stats */
async function updateStats() {
  try {
    const s = await api.system.stats();
    const inf = await api.system.inference();

    // CPU
    const cpu = Math.round(s.cpu_percent);
    document.getElementById('cpu-stat').textContent = cpu + '%';
    document.getElementById('cpu-bar').style.width = cpu + '%';
    document.getElementById('cpu-cores').textContent = s.cpu_count + ' cores';
    document.getElementById('cpu-mini').style.width = cpu + '%';

    // RAM
    const ramPct = Math.round(s.ram_percent);
    document.getElementById('ram-stat').textContent = s.ram_used_gb + ' / ' + s.ram_total_gb + ' GB';
    document.getElementById('ram-bar').style.width = ramPct + '%';
    document.getElementById('ram-pct').textContent = ramPct + '%';
    document.getElementById('ram-mini').style.width = ramPct + '%';

    // GPU
    if (s.gpu) {
      document.getElementById('gpu-card').style.display = 'block';
      document.getElementById('gpu-name').textContent = s.gpu.name || 'GPU';
      const gpuPct = Math.round((s.gpu.memory_used_mb / s.gpu.memory_total_mb) * 100) || 0;
      document.getElementById('gpu-mem').textContent = s.gpu.memory_used_mb + ' / ' + s.gpu.memory_total_mb + ' MB';
      document.getElementById('gpu-bar').style.width = gpuPct + '%';
      document.getElementById('gpu-temp').textContent = (s.gpu.temperature_c || '—') + '°C';
      document.getElementById('gpu-util').textContent = (s.gpu.utilization_pct || '—') + '%';
    } else {
      document.getElementById('gpu-card').style.display = 'none';
    }

    // Inference
    document.getElementById('inf-model').textContent = inf.model_name || '—';
    document.getElementById('inf-status').textContent = inf.is_generating ? '🟢 Generating' : 'Idle';
    document.getElementById('inf-tokens').textContent = inf.total_tokens_generated + ' tokens total';

    // System info
    document.getElementById('uptime').textContent = formatUptime(s.uptime_seconds);
  } catch (e) {
    // Silently fail — server might not be running
  }
}

function formatUptime(s) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h + 'h ' + m + 'm';
}

// Poll every 2 seconds
setInterval(updateStats, 2000);
updateStats();
