// 深度学习基础图表：全部颜色取自 CSS 变量
(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  var baseGrid = { left: 56, right: 28, top: 32, bottom: 44 };

  // ---------- 图 1：学习率调度 ----------
  var elLr = document.getElementById('chart-lr');
  if (elLr && window.echarts) {
    var lr = echarts.init(elLr, null, { renderer: 'svg' });
    var epochs = 150, warm = 3, lrf = 0.01;
    var total = epochs;
    var data = [];
    for (var e = 0; e <= total; e += 1) {
      var v;
      if (e <= warm) {
        v = 0.001 + (0.01 - 0.001) * (e / warm);
      } else {
        var p = (e - warm) / (total - warm);
        v = 0.01 * (1 - p * (1 - lrf)) ;
      }
      data.push([e, +(v * 1000).toFixed(3)]);
    }
    lr.setOption({
      animation: false,
      grid: baseGrid,
      tooltip: { trigger: 'axis', appendToBody: true, valueFormatter: function (v) { return v + 'e-3'; } },
      xAxis: {
        type: 'value', min: 0, max: total,
        name: 'epoch', nameLocation: 'middle', nameGap: 26,
        axisLine: { lineStyle: { color: rule } },
        axisLabel: { color: muted },
        splitLine: { show: false }
      },
      yAxis: {
        type: 'value', name: 'lr',
        axisLine: { show: false },
        axisLabel: { color: muted, formatter: '{value}e-3' },
        splitLine: { lineStyle: { color: rule } }
      },
      series: [{
        type: 'line', data: data, showSymbol: false, smooth: false,
        lineStyle: { color: accent, width: 2.2 },
        areaStyle: { color: accent + '22' },
        markLine: {
          symbol: 'none', silent: true,
          lineStyle: { color: accent2, type: 'dashed', width: 1.2 },
          label: { color: accent2, formatter: 'warmup 结束', position: 'insideEndTop' },
          data: [{ xAxis: warm }]
        }
      }]
    });
    window.addEventListener('resize', function () { lr.resize(); });
  }

  // ---------- 图 2：Adam 二阶矩的有效窗口 ----------
  var elExp = document.getElementById('chart-ema');
  if (elExp && window.echarts) {
    var ema = echarts.init(elExp, null, { renderer: 'svg' });
    var steps = [], b1 = [], b2 = [];
    for (var t = 1; t <= 40; t += 1) {
      steps.push(t);
      b1.push(+((1 - Math.pow(0.9, t)) * 100).toFixed(2));
      b2.push(+((1 - Math.pow(0.999, t)) * 100).toFixed(2));
    }
    ema.setOption({
      animation: false,
      grid: baseGrid,
      legend: { data: ['β₁ = 0.9', 'β₂ = 0.999'], textStyle: { color: muted }, top: 0, right: 0 },
      tooltip: { trigger: 'axis', appendToBody: true, valueFormatter: function (v) { return v + '%'; } },
      xAxis: {
        type: 'category', data: steps,
        name: 'step', nameLocation: 'middle', nameGap: 26,
        axisLine: { lineStyle: { color: rule } },
        axisLabel: { color: muted, interval: 4 },
        splitLine: { show: false }
      },
      yAxis: {
        type: 'value', name: '累积权重 %', max: 100,
        axisLine: { show: false },
        axisLabel: { color: muted },
        splitLine: { lineStyle: { color: rule } }
      },
      series: [
        { name: 'β₁ = 0.9', type: 'line', data: b1, showSymbol: false, lineStyle: { color: accent, width: 2.2 } },
        { name: 'β₂ = 0.999', type: 'line', data: b2, showSymbol: false, lineStyle: { color: accent2, width: 2.2 } }
      ]
    });
    window.addEventListener('resize', function () { ema.resize(); });
  }

  // ---------- 图 3：fp16 与 fp32 的数值差距 ----------
  var elFp = document.getElementById('chart-fp16');
  if (elFp && window.echarts) {
    var fp = echarts.init(elFp, null, { renderer: 'svg' });
    var powers = [];
    for (var p = -8; p <= 8; p += 1) { powers.push(p); }
    var gap = powers.map(function (p) { return +Math.pow(2, p).toPrecision(4); });
    fp.setOption({
      animation: false,
      grid: baseGrid,
      tooltip: { trigger: 'axis', appendToBody: true },
      xAxis: {
        type: 'category', data: powers.map(function (p) { return '2^' + p; }),
        name: '数量级', nameLocation: 'middle', nameGap: 26,
        axisLine: { lineStyle: { color: rule } },
        axisLabel: { color: muted, interval: 2 },
        splitLine: { show: false }
      },
      yAxis: {
        type: 'log', name: '可表示的最小间隔',
        axisLine: { show: false },
        axisLabel: { color: muted },
        splitLine: { lineStyle: { color: rule } }
      },
      series: [{
        type: 'bar', data: gap, barWidth: '52%',
        itemStyle: { color: accent, borderRadius: [2, 2, 0, 0] }
      }]
    });
    window.addEventListener('resize', function () { fp.resize(); });
  }
})();
