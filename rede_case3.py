"""
Rede sintética de 3 barras para ORPD (Optimal Reactive Power Dispatch).

O pandapower não disponibiliza um ``case3`` nativo. Esta rede foi montada
explicitamente para estudos de despacho de reativos, com:

  Barra 0 — Slack (controle de tensão)
  Barra 1 — Gerador PV (controle de tensão + P fixo)
  Barra 2 — Carga PQ

Controles tipicamente otimizados no ORPD:
  - Tensões dos geradores (slack + PV) ............. 2 variáveis contínuas
  - Tap do transformador 0–2 ....................... 1 variável discreta
  - Estágio do shunt capacitivo na barra 2 ......... 1 variável discreta
"""

from __future__ import annotations

import pandapower as pp


def case3_orpd() -> pp.pandapowerNet:
    """
    Constrói e retorna a rede de 3 barras para ORPD.

    Topologia
    ---------
        0 (slack) ---- linha ---- 1 (gerador)
           |                         |
         trafo                     linha
           |                         |
           +--------- 2 (carga) -----+
                         |
                       shunt
    """
    net = pp.create_empty_network(name="case3_orpd", f_hz=60.0, sn_mva=100.0)

    # Tensões nominais iguais facilitam o fluxo em pu e o controle de taps.
    b0 = pp.create_bus(net, vn_kv=230.0, name="Slack")
    b1 = pp.create_bus(net, vn_kv=230.0, name="PV")
    b2 = pp.create_bus(net, vn_kv=230.0, name="Carga")

    # Slack: barra de referência angular e de tensão.
    pp.create_ext_grid(
        net,
        bus=b0,
        vm_pu=1.05,
        va_degree=0.0,
        min_q_mvar=-100.0,
        max_q_mvar=100.0,
        name="Slack",
    )

    # Gerador PV com despacho ativo fixo (ORPD não mexe em P).
    pp.create_gen(
        net,
        bus=b1,
        p_mw=50.0,
        vm_pu=1.04,
        min_q_mvar=-60.0,
        max_q_mvar=60.0,
        min_p_mw=10.0,
        max_p_mw=80.0,
        controllable=True,
        name="G2",
    )

    # Carga PQ dominante — puxa reativo e cria espaço para o ORPD.
    pp.create_load(net, bus=b2, p_mw=80.0, q_mvar=30.0, name="Carga")

    # Linhas 0–1 e 1–2 (impedâncias típicas em base 100 MVA / 230 kV).
    # Z_base = V^2 / S = 230^2 / 100 = 529 Ω.
    # r=0.02 pu → 10.58 Ω; x=0.08 pu → 42.32 Ω; comprimento fictício = 1 km.
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=10.58,
        x_ohm_per_km=42.32,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        name="L01",
    )
    pp.create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=1.0,
        r_ohm_per_km=15.87,
        x_ohm_per_km=52.90,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        name="L12",
    )

    # Transformador 0–2 com comutador (±3 passos de 1,5%).
    # Relação 1:1 na posição neutra; o tap atua no lado HV (barra 0).
    # Em pandapower ≥3, tap_changer_type="Ratio" é obrigatório para o
    # comutador alterar a relação no fluxo de carga (senão TAP≡1 no _ppc).
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b0,
        lv_bus=b2,
        sn_mva=100.0,
        vn_hv_kv=230.0,
        vn_lv_kv=230.0,
        vk_percent=10.0,
        vkr_percent=1.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        tap_side="hv",
        tap_neutral=0,
        tap_min=-3,
        tap_max=3,
        tap_step_percent=1.5,
        tap_pos=0,
        tap_changer_type="Ratio",
        name="T02",
    )

    # Banco capacitivo na barra de carga: 15 MVAr por estágio, on/off.
    # Em pandapower, q_mvar < 0 representa injeção capacitiva.
    pp.create_shunt(
        net,
        bus=b2,
        q_mvar=-15.0,
        p_mw=0.0,
        vn_kv=230.0,
        step=1,
        max_step=1,
        name="Shunt_C2",
    )

    return net


def resumo_rede(net: pp.pandapowerNet) -> None:
    """Imprime a topologia e o resultado do fluxo de carga base."""
    print(f"Rede: {net.name}")
    print(f"  Barras        : {len(net.bus)}")
    print(f"  Slack         : {len(net.ext_grid)}  (barra {int(net.ext_grid.bus.iloc[0])})")
    print(f"  Geradores     : {len(net.gen)}  (barra {int(net.gen.bus.iloc[0])})")
    print(f"  Cargas        : {len(net.load)}  (barra {int(net.load.bus.iloc[0])})")
    print(f"  Linhas        : {len(net.line)}")
    print(f"  Transformador : {len(net.trafo)}  tap_pos={int(net.trafo.tap_pos.iloc[0])}")
    print(f"  Shunt         : {len(net.shunt)}  step={int(net.shunt.step.iloc[0])}")

    pp.runpp(net, numba=False)
    perdas = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
    print("\nFluxo de carga (caso base):")
    print(f"  Perdas ativas : {perdas:.6f} MW")
    print("  Tensões [pu]  :")
    for i, vm in enumerate(net.res_bus.vm_pu):
        print(f"    barra {i}: {vm:.4f}")


if __name__ == "__main__":
    rede = case3_orpd()
    resumo_rede(rede)
