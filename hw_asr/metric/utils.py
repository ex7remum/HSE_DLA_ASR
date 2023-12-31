# Don't forget to support cases when target_text == ''
import editdistance

def calc_cer(target_text: str, predicted_text: str) -> float:
    if not target_text:
        if predicted_text:
            return 1
        return 0
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    if not target_text:
        if predicted_text:
            return 1
        return 0
    target_text_splited = target_text.split(' ')
    pred_text_splited = predicted_text.split(' ')
    return editdistance.eval(target_text_splited, pred_text_splited) / len(target_text_splited)
            