import numpy as np                  # numpy importieren (für spätere Erweiterungen)

class Tracker:
    def __init__(self, iou_threshold = 0.3, max_age = 5):        # Konstruktor: wird beim Erstellen aufgerufen
        self.tracks = {}            # alle bekannten Drohnen: {id: {"bbox": ..., "age": 0}}
        self.next_id = 0            # nächste freie ID für neue Drohne
        self.iou_threshold = iou_threshold  # Mindestüberlappung damit zwei Boxen als gleiche Drohne gelten
        self.max_age = max_age      # wie viele Frames eine Drohne unsichtbar sein darf bevor sie gelöscht wird

    def _iou(self, a, b):                   # private Hilfsmethode: berechnet Überlappung zweier Boxen
        ax1, ay1, ax2, ay2 = a             # Koordinaten von Box A entpacken
        bx1, by1, bx2, by2 = b             # Koordinaten von Box B entpacken
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)  # obere linke Ecke des Überschneidungsrechtecks
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)  # untere rechte Ecke des Überschneidungsrechtecks
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)  # Schnittfläche (0 wenn kein Overlap)
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter  # Gesamtfläche minus Schnitt
        if union > 0:                       # Normalfall
            return inter / union            # IoU = Schnittfläche / Gesamtfläche (0.0 bis 1.0)
        elif union == 0:                    # Sonderfall: beide Boxen haben keine Fläche
            return 0

    def update(self, detections):           # Hauptmethode: wird jeden Frame mit neuen Detektionen aufgerufen
        matched = set()                     # Set: merkt welche Track-IDs in diesem Frame gesehen wurden
        for det in detections:              # für jede neu erkannte Drohne in diesem Frame
            best_id = None                  # beste passende Track-ID (noch keine gefunden)
            best_iou = self.iou_threshold   # Mindestschwelle: nur Überlappung > 30% zählt
            for tid, track in self.tracks.items():              # alle bekannten Tracks durchgehen
                Iou_new = self._iou(det["bbox"], track["bbox"]) # Überlappung mit diesem Track berechnen
                if Iou_new > best_iou:      # besser als bisheriger Kandidat?
                    best_id = tid           # diesen Track als besten Kandidat merken
                    best_iou = Iou_new      # neuen Bestwert speichern
            if best_id is not None:                             # passenden Track gefunden → bekannte Drohne
                self.tracks[best_id]["bbox"] = det["bbox"]      # Position auf neue bbox aktualisieren
                self.tracks[best_id]["age"] = 0                 # Alters-Zähler zurücksetzen
                matched.add(best_id)                            # ID als gesehen markieren
            else:                                               # kein passender Track → neue Drohne
                self.tracks[self.next_id] = {"bbox": det["bbox"], "age": 0}  # neuen Track anlegen
                self.next_id += 1                               # ID-Zähler für nächste neue Drohne erhöhen
        delete = []                         # Liste der IDs die gelöscht werden sollen
        for tid in self.tracks:             # alle bekannten Tracks durchgehen
            if tid not in matched:          # wurde dieser Track in diesem Frame NICHT gesehen?
                self.tracks[tid]["age"] += 1            # Alters-Zähler erhöhen
                if self.tracks[tid]["age"] > self.max_age:  # zu alt → löschen
                    delete.append(tid)
        for tid in delete:                  # jetzt löschen (nicht während der Iteration oben!)
            del self.tracks[tid]            # Track aus dem Dictionary entfernen
        return self.tracks                  # aktuellen Stand aller Tracks zurückgeben
