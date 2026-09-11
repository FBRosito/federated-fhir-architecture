#!/usr/bin/env python3
"""
generate_dataset.py — Synthetic clinical dataset generator (smoke-test only).

For each of the 20 original records in clinical_evolutions.csv, generates
4 synthetic variants totalling 100 examples (20 original + 80 new).
Variants randomise patient demographics, symptoms, vital signs, and clinician
while preserving the same diagnosis (raw_diagnosis) and partition (partition_id)
to maintain the Non-IID specialty distribution across silos.

NOTE: This synthetic dataset is used only for smoke tests. Paper experiments use MIMIC-IV.

Usage:
    uv run python etl_worker/generate_dataset.py \
        --input  etl_worker/data/clinical_evolutions.csv \
        --output etl_worker/data/clinical_evolutions_100.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary data for name and clinician generation (synthetic patients)
# ─────────────────────────────────────────────────────────────────────────────

MALE_FIRST = [
    "Adriano",
    "Alberto",
    "Alexandre",
    "Anderson",
    "André",
    "Bruno",
    "Carlos",
    "Cláudio",
    "Daniel",
    "Diego",
    "Eduardo",
    "Fábio",
    "Felipe",
    "Fernando",
    "Francisco",
    "Gabriel",
    "Guilherme",
    "Gustavo",
    "Hugo",
    "João",
    "Jorge",
    "José",
    "Leonardo",
    "Lucas",
    "Luiz",
    "Marcelo",
    "Márcio",
    "Marcos",
    "Mateus",
    "Miguel",
    "Paulo",
    "Pedro",
    "Rafael",
    "Renato",
    "Ricardo",
    "Roberto",
    "Rodrigo",
    "Sérgio",
    "Thiago",
    "Victor",
]

FEMALE_FIRST = [
    "Adriana",
    "Amanda",
    "Ana",
    "Beatriz",
    "Camila",
    "Carla",
    "Carolina",
    "Clara",
    "Cristina",
    "Daniela",
    "Débora",
    "Eliane",
    "Fabiana",
    "Fernanda",
    "Gabriela",
    "Helena",
    "Isabela",
    "Juliana",
    "Karen",
    "Larissa",
    "Laura",
    "Letícia",
    "Luciana",
    "Marcia",
    "Maria",
    "Mariana",
    "Mônica",
    "Natália",
    "Patricia",
    "Paula",
    "Priscila",
    "Rafaela",
    "Renata",
    "Sandra",
    "Sara",
    "Silvia",
    "Simone",
    "Sonia",
    "Tatiana",
    "Vanessa",
]

LAST_NAMES = [
    "Almeida",
    "Alves",
    "Andrade",
    "Araújo",
    "Barbosa",
    "Barros",
    "Borges",
    "Braga",
    "Campos",
    "Cardoso",
    "Carvalho",
    "Castro",
    "Cavalcanti",
    "Correia",
    "Costa",
    "Cruz",
    "Cunha",
    "Dias",
    "Duarte",
    "Farias",
    "Ferreira",
    "Figueiredo",
    "Fontes",
    "Freitas",
    "Gomes",
    "Gonçalves",
    "Guimarães",
    "Lima",
    "Lopes",
    "Luz",
    "Macedo",
    "Machado",
    "Marques",
    "Martins",
    "Melo",
    "Mendes",
    "Miranda",
    "Monteiro",
    "Moraes",
    "Moreira",
    "Nascimento",
    "Nunes",
    "Oliveira",
    "Pereira",
    "Pinto",
    "Ramos",
    "Reis",
    "Ribeiro",
    "Rocha",
    "Rodrigues",
    "Santos",
    "Silva",
    "Soares",
    "Sousa",
    "Souza",
    "Tavares",
    "Teixeira",
    "Torres",
    "Vasconcelos",
    "Vieira",
]

PRACTITIONERS_CARDIOLOGY = [
    "Dra. Ana Cardoso",
    "Dr. Roberto Lima",
    "Dra. Luciana Prado",
    "Dr. Marcos Alves",
    "Dra. Patricia Mendes",
    "Dr. Eduardo Fonseca",
    "Dra. Juliana Castro",
    "Dr. Renato Barbosa",
    "Dra. Carolina Neves",
    "Dr. Alexandre Duarte",
]
PRACTITIONERS_PNEUMOLOGY = [
    "Dr. Paulo Nogueira",
    "Dra. Camila Torres",
    "Dr. Henrique Barros",
    "Dra. Vanessa Campos",
    "Dr. Leandro Farias",
    "Dra. Fernanda Cruz",
    "Dr. Gustavo Mendes",
    "Dra. Tatiana Rocha",
    "Dr. Ricardo Monteiro",
    "Dra. Simone Lopes",
]
PRACTITIONERS_ENDOCRINOLOGY = [
    "Dra. Renata Silveira",
    "Dr. Sergio Nascimento",
    "Dra. Isabela Rocha",
    "Dr. Carlos Brandão",
    "Dra. Fernanda Dias",
    "Dr. Flávio Correia",
    "Dra. Mariana Teixeira",
    "Dr. Paulo Gomes",
    "Dra. Cristina Alves",
    "Dr. André Cavalcanti",
]
PRACTITIONERS_GENERAL = [
    "Dr. Thiago Moreira",
    "Dra. Monica Leal",
    "Dr. Wagner Fonseca",
    "Dra. Adriana Costa",
    "Dr. Bruno Carvalho",
    "Dra. Sandra Freitas",
    "Dr. Leonardo Borges",
    "Dra. Daniela Pereira",
    "Dr. Felipe Santos",
    "Dra. Priscila Oliveira",
]

PRACTITIONERS_BY_PARTITION = {
    0: PRACTITIONERS_CARDIOLOGY,
    1: PRACTITIONERS_PNEUMOLOGY,
    2: PRACTITIONERS_ENDOCRINOLOGY,
    3: PRACTITIONERS_GENERAL,
}

# ─────────────────────────────────────────────────────────────────────────────
# Clinical text variants per diagnosis.
# Each list contains 4 template strings for the 4 synthetic examples.
# {age}, {gender_adj}, {gender_pron} are substituted at runtime.
# ─────────────────────────────────────────────────────────────────────────────

CLINICAL_VARIANTS: dict[str, list[str]] = {
    "Hipertensão arterial sistêmica": [
        "Paciente do sexo {gender_noun}, {age} anos, com queixa de cefaleia occipital e visão turva há 5 dias. PA: 178/108 mmHg. FC: 78 bpm. Retinoscopia: cruzamentos arteriovenosos. Ajustado esquema anti-hipertensivo com losartana 100 mg. Orientado sobre adesão e dieta.",
        "Paciente do sexo {gender_noun}, {age} anos, chegou ao pronto-atendimento com PA: 190/115 mmHg após estresse emocional intenso. Sem lesão de órgão-alvo aguda. Administrado captopril sublingual com boa resposta. Alta com ajuste do anti-hipertensivo habitual.",
        "Paciente do sexo {gender_noun}, {age} anos, em seguimento ambulatorial de HAS há 8 anos. PA mal controlada: 165/102 mmHg mesmo com 2 medicamentos. Adicionado anlodipino 5 mg ao esquema. Solicitados ecocardiograma e microalbuminúria.",
        "Paciente do sexo {gender_noun}, {age} anos, com hipertensão de difícil controle. Monitorização ambulatorial (MAPA): carga hipertensiva diurna 72%. Investigação de causas secundárias negativa. Introduzida espironolactona como 4.º fármaco.",
    ],
    "Insuficiência cardíaca congestiva": [
        "Paciente do sexo {gender_noun}, {age} anos, com piora da classe funcional (NYHA III) e ganho de 4 kg em 1 semana. BNP: 890 pg/mL. Ecocardiograma: FE 32%. Otimização de furosemida e carvedilol. Orientação sobre restrição hídrica.",
        "Paciente do sexo {gender_noun}, {age} anos, internado por descompensação de IC com ortopneia e crepitações bibasais. Natriurese em queda. Introduzida dobutamina EV. Ecocardiograma: FE 28%, dilatação biventricular.",
        "Paciente do sexo {gender_noun}, {age} anos, com IC de etiologia isquêmica, FE 38%. Revisão de medicamentos: adicionado dapagliflozina. Monitoração de função renal. Encaminhado para programa de reabilitação cardíaca.",
        "Paciente do sexo {gender_noun}, {age} anos, em seguimento pós-hospitalização por IC. Ecocardiograma de controle: FE melhorou de 25% para 35% com otimização do tratamento. Mantida terapia quádrupla (IECA, betabloqueador, MRA, SGLT2i).",
    ],
    "Angina instável": [
        "Paciente do sexo {gender_noun}, {age} anos, com dor precordial em aperto ao repouso há 3 horas. ECG: infradesnivelamento de ST em V1-V3. Troponina ultrassensível 2× o limite normal. Admitido com dupla antiagregação e anticoagulação para cateterismo de urgência.",
        "Paciente do sexo {gender_noun}, {age} anos, com angina instável de início recente (crescente). Cintilografia miocárdica: hipoperfusão anterior. TIMI score: 4. Encaminhado para coronariografia — lesão de 70% em DA proximal. Indicado stent farmacológico.",
        "Paciente do sexo {gender_noun}, {age} anos, com dor torácica noturna repetida nos últimos 4 dias. ECG basal normal. Troponina seriada negativa. Teste ergométrico positivo para isquemia. Cinecoronariografia indicada.",
        "Paciente do sexo {gender_noun}, {age} anos, diabético, com equivalente anginoso (dispneia de esforço). Holter: alterações de ST transitórias. Cortiça: lesão de tronco 50%. Encaminhado para cirurgia de revascularização miocárdica.",
    ],
    "Infarto agudo do miocárdio": [
        "Paciente do sexo {gender_noun}, {age} anos, com dor retroesternal intensa e diaforese há 45 minutos. ECG: supradesnivelamento de ST em V1-V4. Troponina I: 12,4 ng/mL. ICP primária realizada com stent em DA; TIMI 3 ao final.",
        "Paciente do sexo {gender_noun}, {age} anos, com IAM inferior (ST supra em DII, DIII, aVF). Admitido em 90 min do início dos sintomas. Hemodinâmica: oclusão total de CD. Angioplastia com sucesso; fração de ejeção pós-procedimento: 52%.",
        "Paciente do sexo {gender_noun}, {age} anos, com IAM sem supradesnivelamento de ST (NSTEMI) de alto risco. Troponina: 18 ng/mL. ICP em 24 horas: lesão crítica em CX. Alta com dupla antiagregação por 12 meses.",
        "Paciente do sexo {gender_noun}, {age} anos, com IAM anterior extenso complicado por choque cardiogênico. IABP instalado. ICP em DA e lesão de tronco. Admitido em UTI cardiológica para monitoração intensiva.",
    ],
    "AVC isquêmico": [
        "Paciente do sexo {gender_noun}, {age} anos, com afasia súbita e hemiplegia esquerda. NIHSS: 16. TC: hiperdensidade em ACM direita. Submetido a trombectomia mecânica com recanalização completa (TICI 2b). Transferido para unidade de AVC.",
        "Paciente do sexo {gender_noun}, {age} anos, com AVC isquêmico lacunar (cápsula interna esquerda) evidenciado em RNM-DWI. NIHSS: 6. Fora da janela para trombólise. Iniciada antiagregação e estatina de alta potência. Fisioterapia iniciada em 24 h.",
        "Paciente do sexo {gender_noun}, {age} anos, com AIT de 30 minutos (amaurose fugaz esquerda). ABCD² score: 5. RNM: sem lesão aguda. Ecocardiograma: forame oval patente. Dupla antiagregação transitória; avaliação para fechamento percutâneo.",
        "Paciente do sexo {gender_noun}, {age} anos, com AVC isquêmico cardioembólico (FA paroxística detectada no Holter). Anticoagulação com apixabana iniciada após 48 h. Reabilitação multidisciplinar iniciada precocemente.",
    ],
    "DPOC exacerbado": [
        "Paciente do sexo {gender_noun}, {age} anos, ex-tabagista (30 maços-ano), com piora do broncoespasmo e febre. SpO2: 85% em AA. Espirometria prévia: VEF1/CVF 0,58 (GOLD 3). Iniciado prednisona, salbutamol nebulizado e azitromicina. Internado para suporte de O2.",
        "Paciente do sexo {gender_noun}, {age} anos, DPOC GOLD 2, com aumento da dispneia e escarro amarelado há 3 dias. CRP: 18 mg/dL. Radiografia: hiperinsuflação sem consolidação. Amoxicilina + corticosteroide oral por 5 dias. Internação evitada.",
        "Paciente do sexo {gender_noun}, {age} anos, em uso de VNI domiciliar, admitido com hipercapnia aguda (pCO2: 68 mmHg). VNI hospitalar com boa resposta. Broncodilatadores inalatórios otimizados. Encaminhado para programa de reabilitação pulmonar.",
        "Paciente do sexo {gender_noun}, {age} anos, com DPOC grave e 3 exacerbações no ano. Cultura de escarro: Pseudomonas aeruginosa. Iniciado ciprofloxacino IV. Avaliação para elegibilidade à cirurgia redutora de volume.",
    ],
    "Asma brônquica": [
        "Paciente do sexo {gender_noun}, {age} anos, com crise asmática grave (fala em palavras). SpO2: 88%. Peak flow: 35% previsto. Três nebulizações, corticosteroide IV e sulfato de magnésio. Internado para monitoração. Alta após 24 h com escalonamento do tratamento.",
        "Paciente do sexo {gender_noun}, {age} anos, com asma parcialmente controlada apesar de CI/LABA em dose média. Teste de IgE específica: alérgeno a ácaro. Encaminhado para imunoterapia. Técnica inalatória corrigida na consulta.",
        "Paciente do sexo {gender_noun}, {age} anos, com asma de difícil controle (step 4). Eosinófilos: 520/µL. FeNO: 45 ppb. Indicado biológico (mepolizumabe). Orientado sobre identificação de gatilhos ambientais.",
        "Paciente do sexo {gender_noun}, {age} anos, com broncoespasmo desencadeado por AAS (asma aspirínica). SpO2: 91%. Atendido com nebulização e corticosteroide. Prescrição de leukotriene antagonista. Lista de AINEs evitar entregue ao paciente.",
    ],
    "Pneumonia": [
        "Paciente do sexo {gender_noun}, {age} anos, com febre alta (39,5°C), tosse com expectoração esverdeada e dispneia moderada há 5 dias. SatO2: 91%. Radiografia: consolidação bilobar esquerda. PSI IV. Internado com levofloxacino EV.",
        "Paciente do sexo {gender_noun}, {age} anos, com pneumonia adquirida na comunidade de etiologia atípica (Mycoplasma). Quadro subagudo com cefaleia e tosse seca. IgM Mycoplasma reagente. Claritromicina oral por 14 dias. Boa evolução ambulatorial.",
        "Paciente do sexo {gender_noun}, {age} anos, imunossuprimido por uso de corticosteroide crônico, com febre e infiltrado intersticial bilateral. Lavado broncoalveolar: Pneumocystis jirovecii. Iniciado sulfametoxazol-trimetoprim em dose plena.",
        "Paciente do sexo {gender_noun}, {age} anos, com pneumonia associada à assistência à saúde (PAAS). Hemocultura: K. pneumoniae ESBL. IPMN-cilastatin iniciado após antibiograma. UTI por insuficiência respiratória progressiva.",
    ],
    "Derrame pleural": [
        "Paciente do sexo {gender_noun}, {age} anos, com derrame pleural à esquerda detectado em tomografia. Toracocentese: exsudato com LDH 3× LSN, pH 7,18. Suspeita de empiema. Drenagem pleural instalada. Cultura positiva para Streptococcus milleri.",
        "Paciente do sexo {gender_noun}, {age} anos, com derrame pleural bilateral pequeno em contexto de IC descompensada. Transudato (critérios de Light). Resposta a diuréticos. Ecocardiograma: FE 30%.",
        "Paciente do sexo {gender_noun}, {age} anos, com derrame pleural maligno confirmado (adenocarcinoma de pulmão). Pleurodese química com talco realizada via pleuroscopia. Oncologia acionada para estadiamento.",
        "Paciente do sexo {gender_noun}, {age} anos, com derrame pleural recorrente após 2 toracocenteses. Citologia oncótica positiva. Cateter pleural permanente instalado para drenagem intermitente domiciliar.",
    ],
    "Tromboembolismo pulmonar": [
        "Paciente do sexo {gender_noun}, {age} anos, com TEP maciço e instabilidade hemodinâmica. Ecocardiograma: disfunção de VD e McConnel sign. Trombolítico sistêmico (alteplase 100 mg) administrado com recuperação hemodinâmica.",
        "Paciente do sexo {gender_noun}, {age} anos, com TEP de risco intermediário-alto. BNP: 450 pg/mL. Troponina elevada. Score PESI III. Internado em UTI com anticoagulação plena e monitoração contínua.",
        "Paciente do sexo {gender_noun}, {age} anos, com 1.º episódio de TEP provocado (imobilização prolongada). Angio-TC: defeito de preenchimento bilateral em ramos lobares. Rivaroxabana 15 mg 2×/dia por 21 dias, seguido de 20 mg/dia.",
        "Paciente do sexo {gender_noun}, {age} anos, com TEP recorrente mesmo em anticoagulação. Investigação de trombofilia: síndrome antifosfolipídio. Mantida anticoagulação indefinida com warfarina (INR-alvo 2-3).",
    ],
    "Diabetes mellitus tipo 2": [
        "Paciente do sexo {gender_noun}, {age} anos, com DM2 de diagnóstico recente. HbA1c: 11,8%. Sem cetoacidose. Metformina + empagliflozina introduzidos. Encaminhado para programa de educação em diabetes e nutricionista.",
        "Paciente do sexo {gender_noun}, {age} anos, com DM2 descompensado (HbA1c 9,5%) em uso de sulfonilurea isolada. Adicionado inibidor de DPP-4. Rastreio de complicações: retinopatia leve detectada. Encaminhado à oftalmologia.",
        "Paciente do sexo {gender_noun}, {age} anos, com DM2 e doença cardiovascular estabelecida (IAM prévio). Introduzido semaglutida 1 mg semanal. HbA1c meta < 7%. Dieta hipocalórica reforçada.",
        "Paciente do sexo {gender_noun}, {age} anos, com DM2 e nefropatia (TFGe 45 mL/min, microalbuminúria). Metformina mantida em dose reduzida. IECA otimizado. Dapagliflozina com indicação nefroprotetora introduzida.",
    ],
    "Hipotireoidismo": [
        "Paciente do sexo {gender_noun}, {age} anos, com hipotireoidismo de Hashimoto. TSH: 24 mUI/L. Anti-TPO: 1200 UI/mL. Levotiroxina 75 mcg iniciada com titulação progressiva. Retorno laboratorial em 8 semanas.",
        "Paciente do sexo {gender_noun}, {age} anos, com hipotireoidismo subclínico persistente (TSH 8-10 mUI/L em 2 dosagens). Sintomas leves. Iniciada levotiroxina 25 mcg após discussão de riscos e benefícios com paciente.",
        "Paciente do sexo {gender_noun}, {age} anos, pós-tireoidectomia total por carcinoma, em uso de levotiroxina supressiva (TSH alvo < 0,1). TSH atual: 0,08. Dose mantida. Densitometria óssea anual solicitada.",
        "Paciente do sexo {gender_noun}, {age} anos, com hipotireoidismo grave (mixedema). TSH: 45 mUI/L. T4 livre: 0,3 ng/dL. Internada para reposição cautelosa de levotiroxina EV. Hidrocortisona profilática administrada.",
    ],
    "Síndrome metabólica": [
        "Paciente do sexo {gender_noun}, {age} anos, com síndrome metabólica (obesidade abdominal, hipertrigliceridemia, HDL baixo, hiperglicemia leve, HAS). Iniciado programa de estilo de vida intensivo. Estatina introduzida pelo risco cardiovascular aumentado.",
        "Paciente do sexo {gender_noun}, {age} anos, com síndrome metabólica e esteatohepatite não alcoólica (NASH). ALT 3× LSN, USG: fígado hiperecogênico. Perda de 7% do peso indicada como alvo terapêutico principal.",
        "Paciente do sexo {gender_noun}, {age} anos, com síndrome metabólica e apneia obstrutiva do sono (AHI: 28/h). CPAP introduzido. Perda de peso orientada como terapia concomitante. Reavaliação polissonográfica em 6 meses.",
        "Paciente do sexo {gender_noun}, {age} anos, com síndrome metabólica grave (cintura 118 cm, TG 380 mg/dL). Avaliação para cirurgia bariátrica iniciada. Score de Framingham: risco cardiovascular 18% em 10 anos.",
    ],
    "Síndrome dos ovários policísticos": [
        "Paciente do sexo {gender_noun}, {age} anos, com SOP e infertilidade há 2 anos. AMH: 8,2 ng/mL. USG: 16 folículos/ovário. Indução de ovulação com letrozol. Encaminhada para reprodução assistida.",
        "Paciente do sexo {gender_noun}, {age} anos, com SOP e hiperandrogenismo clínico (acne severa, alopecia androgenética). DHEA-S elevada. Anticoncepcional oral com ciproterona iniciado. Dermatologia acionada.",
        "Paciente do sexo {gender_noun}, {age} anos, com SOP e resistência insulínica severa (HOMA-IR: 4,8). Metformina 2 g/dia. Orientação nutricional com foco em índice glicêmico. Risco aumentado para DM2 discutido.",
        "Paciente do sexo {gender_noun}, {age} anos, adolescente com SOP (critérios de NIH). Ciclos irregulares há 3 anos. Ultrassonografia: morfologia policística. Anticoncepcional oral introduzido; acompanhamento multidisciplinar.",
    ],
    "Hiperparatireoidismo primário": [
        "Paciente do sexo {gender_noun}, {age} anos, com hipercalcemia sintomática (cálcio: 12,8 mg/dL): nefrolitíase recorrente e fraqueza muscular. PTH: 210 pg/mL. Cintilografia: adenoma de paratireoide direito. Cirurgia indicada.",
        "Paciente do sexo {gender_noun}, {age} anos, com hiperparatireoidismo primário e osteoporose severa (T-score coluna: -2,9). Paratireoidectomia realizada. DXA de controle programada para 1 ano. Suplementação de vitamina D e cálcio.",
        "Paciente do sexo {gender_noun}, {age} anos, com hiperparatireoidismo primário assintomático em observação. Critérios cirúrgicos não atingidos. Monitoração anual: calcemia, PTH, DXA e USG renal. Hidratação adequada reforçada.",
        "Paciente do sexo {gender_noun}, {age} anos, com hiperparatireoidismo primário e crise hipercalcêmica (cálcio: 14,2 mg/dL). Hidratação vigorosa EV, furosemida e zoledronato. Paratireoidectomia de urgência após estabilização.",
    ],
    "Lombalgia aguda": [
        "Paciente do sexo {gender_noun}, {age} anos, com lombalgia aguda mecânica após levantamento de peso. Lasègue positivo à esquerda. Irradiação para o membro inferior esquerdo (L4-L5). RNM: protrusão discal. Fisioterapia e analgesia otimizada.",
        "Paciente do sexo {gender_noun}, {age} anos, com lombalgia aguda sem sinais de alerta (febre ou perda de peso). Sem déficit neurológico. Orientado sobre atividade física progressiva. Naproxeno e ciclobenzaprina por 5 dias.",
        "Paciente do sexo {gender_noun}, {age} anos, com lombalgia aguda em contexto de trabalho físico repetitivo. Avaliação ergonômica solicitada. Afastamento médico de 7 dias. Encaminhado para fisioterapia ocupacional.",
        "Paciente do sexo {gender_noun}, {age} anos, com lombalgia aguda recidivante (4.ª crise em 1 ano). RNM: espondiloartrose L3-S1 sem compressão radicular. Pilates terapêutico indicado como prevenção de novas crises.",
    ],
    "Infecção do trato urinário": [
        "Paciente do sexo {gender_noun}, {age} anos, com ITU complicada (pielonefrite). Febre 39°C, dor lombar bilateral, calafrios. Hemocultura e urinocultura coletadas. Ceftriaxona EV iniciada. Internação para hidratação e antibioticoterapia parenteral.",
        "Paciente do sexo {gender_noun}, {age} anos, com ITU de repetição (3.ª em 6 meses). Cultura: E. coli ESBL. Antibioticoterapia guiada por antibiograma (meropenem oral). Investigação urológica com USG renal solicitada.",
        "Paciente do sexo {gender_noun}, {age} anos, com cistite aguda não complicada. Disúria e polaciúria há 2 dias. EAS: leucocitúria e nitrito +. Fosfomicina 3 g dose única prescrita. Orientada sobre hidratação e higiene.",
        "Paciente do sexo {gender_noun}, {age} anos, com ITU em usuário de cateter vesical. Urinocultura: Klebsiella pneumoniae. Antibioticoterapia de 7 dias. Reavaliação da necessidade do cateter; removido após melhora clínica.",
    ],
    "Artrite reumatoide": [
        "Paciente do sexo {gender_noun}, {age} anos, com AR de início recente (< 1 ano). FR: 1:80. Anti-CCP: 150 U/mL. Iniciado metotrexato 15 mg/semana + ácido fólico. Meta: remissão ou baixa atividade em 6 meses (treat-to-target).",
        "Paciente do sexo {gender_noun}, {age} anos, com AR soropositiva em atividade moderada (DAS28: 4,2) apesar de metotrexato em dose plena. Introduzido adalimumabe. Rastreio de tuberculose negativo. Vacinas atualizadas.",
        "Paciente do sexo {gender_noun}, {age} anos, com AR e manifestação extra-articular: nódulos reumatoides e síndrome de Sjögren secundária. Hidroxicloroquina adicionada. Avaliação oftalmológica anual recomendada.",
        "Paciente do sexo {gender_noun}, {age} anos, com AR em remissão (DAS28: 1,8) por 18 meses. Tentativa de desmame gradual do biológico iniciada. Monitoração com ultrassonografia articular trimestral.",
    ],
    "Depressão": [
        "Paciente do sexo {gender_noun}, {age} anos, com episódio depressivo maior (PHQ-9: 21). Hospitalização voluntária por risco aumentado de suicídio. Sertralina 100 mg e olanzapina 5 mg iniciados. Psicoterapia intensiva no hospital.",
        "Paciente do sexo {gender_noun}, {age} anos, com depressão resistente (2 ensaios terapêuticos sem resposta). Encaminhado para avaliação de estimulação magnética transcraniana (EMT). Psiquiatra ajusta para venlafaxina + bupropiona.",
        "Paciente do sexo {gender_noun}, {age} anos, com depressão pós-parto (Edinburg: 16). Parceiro incluído no processo. Sertralina iniciada com segurança na lactação. Psicoterapia cognitivo-comportamental semanal.",
        "Paciente do sexo {gender_noun}, {age} anos, com depressão leve-moderada (PHQ-9: 11). Optado por psicoterapia como 1.ª linha. Atividade física prescrita. Reavaliação em 4 semanas; farmacoterapia reservada se sem melhora.",
    ],
    "Doença renal crônica": [
        "Paciente do sexo {gender_noun}, {age} anos, com DRC estadio 4 (TFGe 18 mL/min) por glomerulonefrite lúpica. Biópsia renal: nefrite classe IV. Micofenolato mofetil e corticosteroide introduzidos. Diálise em preparo.",
        "Paciente do sexo {gender_noun}, {age} anos, com DRC estadio 3b em diabético. Fistula arteriovenosa confeccionada para acesso para hemodiálise futuro. Consulta com nefrologia mensal. Dieta hipoproteica orientada.",
        "Paciente do sexo {gender_noun}, {age} anos, com DRC por nefropatia hipertensiva. Creatinina: 2,2 mg/dL. Anemia tratada com eritropoetina. Carbonato de cálcio prescrito para hiperfosfatemia. PA alvo < 130/80 mmHg.",
        "Paciente do sexo {gender_noun}, {age} anos, em hemodiálise há 2 anos por DRC terminal. Lista de transplante ativada. Controle do hiperparatireoidismo secundário com cinacalcete. Avaliação cardiovascular pré-transplante solicitada.",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def random_name(gender: str, rng: random.Random) -> str:
    """Generate a random Brazilian-style full name (first + two distinct surnames)."""
    first = rng.choice(MALE_FIRST if gender == "male" else FEMALE_FIRST)
    last1 = rng.choice(LAST_NAMES)
    last2 = rng.choice(LAST_NAMES)
    while last2 == last1:
        last2 = rng.choice(LAST_NAMES)
    return f"{first} {last1} {last2}"


def random_birth_date(age: int, record_date_str: str, rng: random.Random) -> str:
    """Returns a birth date consistent with the given age."""
    record_year = int(record_date_str[:4])
    birth_year = record_year - age
    birth_month = rng.randint(1, 12)
    birth_day = rng.randint(1, 28)
    return f"{birth_year}-{birth_month:02d}-{birth_day:02d}"


def vary_record_date(original: str, variant_index: int) -> str:
    """Shifts the encounter date so each variant has a unique timestamp."""
    base = date.fromisoformat(original[:10])
    offset = (variant_index + 1) * 7  # subsequent weeks
    new_date = base + timedelta(days=offset)
    return (
        f"{new_date.isoformat()}T{original[11:] if len(original) > 10 else '08:00:00Z'}"
    )


def next_patient_id(counter: int) -> str:
    """Format a zero-padded synthetic patient ID (``P001``, ``P002``, ...)."""
    return f"P{counter:03d}"


# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────


def generate_variants(
    original: dict[str, Any],
    variant_index: int,
    patient_counter: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Generates a single synthetic example from an original record."""
    diagnosis = original["raw_diagnosis"]
    partition_id = int(original["partition_id"])

    # Alternate gender across variant pairs for diversity
    gender = (
        "male"
        if (variant_index % 2 == 0) == (original["gender"] == "female")
        else "female"
    )

    # Age band varies per variant (young, middle-aged, young-elderly, elderly)
    age_bands = [32, 45, 60, 74]
    age = age_bands[variant_index] + rng.randint(-4, 4)

    gender_noun = "masculino" if gender == "male" else "feminino"

    templates = CLINICAL_VARIANTS.get(diagnosis, [])
    if len(templates) > variant_index:
        clinical_text = templates[variant_index].format(
            gender_noun=gender_noun,
            age=age,
        )
    else:
        # Generic fallback when the diagnosis has no template for this variant
        clinical_text = (
            f"Paciente do sexo {gender_noun}, {age} anos, com {diagnosis}. "
            "Avaliação clínica realizada, conduta terapêutica otimizada."
        )

    practitioners = PRACTITIONERS_BY_PARTITION.get(partition_id, PRACTITIONERS_GENERAL)
    practitioner = rng.choice(
        [p for p in practitioners if p != original["practitioner"]] or practitioners
    )

    return {
        "patient_id": next_patient_id(patient_counter),
        "patient_name": random_name(gender, rng),
        "gender": gender,
        "birth_date": random_birth_date(age, original["record_date"], rng),
        "record_date": vary_record_date(original["record_date"], variant_index),
        "raw_diagnosis": diagnosis,
        "clinical_text": clinical_text,
        "practitioner": practitioner,
        "partition_id": original["partition_id"],
        "partition_label": original["partition_label"],
    }


def main() -> None:
    """CLI entry point: build the expanded synthetic clinical dataset."""
    parser = argparse.ArgumentParser(
        description="Generate the expanded synthetic clinical dataset."
    )
    parser.add_argument(
        "--input",
        default="etl_worker/data/clinical_evolutions.csv",
        help="CSV de entrada com os 20 exemplos originais.",
    )
    parser.add_argument(
        "--output",
        default="etl_worker/data/clinical_evolutions_100.csv",
        help="CSV de saída com 100 exemplos (20 originais + 80 sintéticos).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente para reproducibilidade (padrão: 42).",
    )
    parser.add_argument(
        "--variants-per-original",
        type=int,
        default=4,
        help="Número de variantes a gerar por exemplo original (padrão: 4).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "patient_id",
        "patient_name",
        "gender",
        "birth_date",
        "record_date",
        "raw_diagnosis",
        "clinical_text",
        "practitioner",
        "partition_id",
        "partition_label",
    ]

    originals: list[dict[str, Any]] = []
    with input_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            originals.append(dict(row))

    patient_counter = len(originals) + 1  # P021, P022, …

    all_rows: list[dict[str, Any]] = list(originals)

    for original in originals:
        for vi in range(args.variants_per_original):
            variant = generate_variants(original, vi, patient_counter, rng)
            all_rows.append(variant)
            patient_counter += 1

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total = len(all_rows)
    synthetic = total - len(originals)
    print(
        f"Dataset generated: {total} examples "
        f"({len(originals)} original + {synthetic} synthetic) → {output_path}"
    )

    # Per-partition summary
    from collections import Counter

    by_partition: Counter = Counter(r["partition_label"] for r in all_rows)
    for label, count in sorted(by_partition.items()):
        print(f"  {label}: {count} examples")


if __name__ == "__main__":
    main()
