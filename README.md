Immune Reactor

Immune Reactor to zaawansowany, modułowy system detekcji anomalii i atestacji kryptograficznej zaprojektowany dla rojów autonomicznych (dronów). System łączy rygorystyczne uwierzytelnianie typu zero-trust z wielodomenową analizą behawioralną opartą na głębokim uczeniu (AI).

Kluczowe cechy
Bezpieczeństwo typu Zero-Trust: Każda jednostka przechodzi weryfikację kryptograficzną (nonce, timestamp, cyfrowy podpis) przed dopuszczeniem do sieci.

Fizykocentryczna analiza AI: Wykorzystuje wyspecjalizowane sub-encodery (Terrain, Motion, Air, MagRF) do tworzenia precyzyjnych odcisków palców stanu drona.

Wykrywanie dryftu (EMA): System dynamicznie uczy się "normalnego" profilu każdego drona, wykrywając anomalie (jak np. jamming GPS) w czasie rzeczywistym.

System Kwarantanny: Automatyczna izolacja jednostek wykazujących zachowania odbiegające od normy, zapobiegająca skażeniu roju.

Architektura
Projekt opiera się na separacji warstw:

src/immunereactor/core/ – Rdzeń analityczny (enkodery, identyfikacja, grafy ryzyka).

src/immunereactor/runtime/ – Silnik wykonawczy zarządzający cyklem życia roju i konsensusem.

src/immunereactor/data/ – Moduły ładowania logów i symulacji strumieni danych.

Instalacja i uruchomienie
Wymagania: Python 3.10+ oraz pip.

Instalacja:

Bash
pip install -e .
Uruchomienie symulacji:

Bash
py -m immunereactor.main
Testy jednostkowe:

Bash
pytest
