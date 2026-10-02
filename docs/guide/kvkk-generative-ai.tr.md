# KVKK ve üretken yapay zekâ

!!! warning "Hukuki danışmanlık değildir"
    Bu sayfa, bir dil modeline metin gönderirken wardcat'in neyi değiştirip
    neyi değiştirmediğini anlatır. Hukuki danışmanlık değildir ve hiçbir
    sistemi 6698 sayılı Kanun ya da başka bir düzenleme karşısında güvence
    altına almaz. Kararı veri koruma sorumlunuz veya hukuk danışmanınızla
    birlikte verin. [English version](kvkk-generative-ai.md).

## Kurumun söyledikleri

**Yurt dışına aktarım.** Kişisel verilerin yurt dışına aktarımı 6698 sayılı
Kanun'un 9'uncu maddesinde düzenlenir. Madde, Mart 2024'te 7499 sayılı Kanun'la
yeniden yazıldı ve 1 Haziran 2024'te yürürlüğe girdi. Aktarım artık bir
yeterlilik kararına, uygun güvencelere (standart sözleşme, bağlayıcı şirket
kuralları gibi) ya da maddede sayılan arızi hallerden birine dayanır. Usul,
10 Temmuz 2024 tarihli ve 32598 sayılı Resmî Gazete'de yayımlanan yönetmelikle
belirlenmiştir. Yeterlilik kararları dahil güncel durum için Kurum'un
[yurt dışına aktarım sayfasına](https://www.kvkk.gov.tr/Icerik/2053/Yurtdisina-Aktarim) bakın.

**Üretken yapay zekâ.** Kurum, *Üretken Yapay Zekâ ve Kişisel Verilerin
Korunması Rehberi (15 Soruda)*'yı 24 Kasım 2025'te yayımladı
([sayfa](https://www.kvkk.gov.tr/Icerik/8547/uretken-yapay-zeka-ve-kisisel-verilerin-korunmasi-rehberi-15-soruda)).
Bu sayfayı ilgilendiren iki noktası var:

- **10. soru.** Türkiye'de faaliyet gösteren bir veri sorumlusu, yurt dışında
  yerleşik bir hizmet sağlayıcı üzerinden üretken yapay zekâ kullanıyor ve bu
  yolla kişisel veri yurt dışına gidiyorsa, aktarımın 9'uncu maddeye ve 2024
  yönetmeliğine uygun yapılması gerekir.
- **Anonim veri.** Anonim ya da anonim hâle getirilmiş veri kişisel veri
  değildir ve Kanun kapsamı dışındadır. Anonim hâle getirildiği ileri sürülen
  verinin gerçekten anonim olup olmadığı teknik yöntemler ve nesnel ölçütlerle
  ortaya konmalıdır. Veri kümesi anonim hâle getirilene kadar kişisel veri
  niteliğini korur.

Rehber takma adlandırmaya (pseudonymisation) değinmiyor.

## wardcat'in değiştirdikleri

- **Taramanın kendisi aktarım değildir.** wardcat sizin sürecinizde çalışır.
  LLM katmanı kendi donanımınızdaki bir modele bağlıysa, taranmak için hiçbir
  metin ağınızdan çıkmaz. Türkiye dışında barındırılan bir bulut PII servisi
  ise taramanın kendisini bir aktarıma çevirir.
- **Gönderdiğiniz metinde daha az tanımlayıcı kalır.** wardcat'in bulup
  değiştirdiği her değer, sağlayıcıya giden istemde yer almaz.

## Değiştirmedikleri

- **Metnin geri kalanı yine yurt dışına gider.** wardcat'in bulamadığı ve
  `warn` eylemindeki her şey sağlayıcıya yazıldığı gibi ulaşır. Tespit en iyi
  çaba esaslıdır: isimler, adresler ve serbest metindeki ayrıntılar belli bir
  oranda kaçar. Bkz. [bilinen sınırlamalar](limitations.md).
- **Token, hash ve surrogate takma addır, anonimleştirme değildir.** Geri
  alınabilir eylemler değerlerin geri gelebilmesi için vardır. Tuzu ya da token
  haritasını elinde tutan metni yeniden kimliklendirebilir. Düşük entropili bir
  değerin (telefon, TC numarası) deterministik hash'i kaba kuvvetle çözülebilir.
  Token'lanmış bir istemi, aksini Kurum'un ölçütüyle gösteremedikçe kişisel veri
  sayın. wardcat bunu kanıtlamaz.
- **Bağlam kimliği ele verir.** "Bursa'da halka açık bir bankanın 1971
  doğumlu finans direktörü" isim geçmeden bir kişiyi tarif eder. Tanımlayıcıları
  silmek bunu ortadan kaldırmaz.
- **Gerçekçi bir surrogate gerçek bir kişiye ait olabilir.** Surrogate TC
  numarası checksum'dan geçer, surrogate isim sıradan bir isimdir. İkisi de
  birine ait olabilir. Bkz. [güvenlik](security.md#surrogates).

## Başlangıç yapılandırması

```python
import os
from wardcat import Wardcat, Action

guard = (
    Wardcat(salt=os.environ["WARDCAT_SALT"])
    .with_preset("kvkk")        # bir başlangıç politikası; uyum iddiası değil
    .with_ner(language="tr")
    .with_strict()              # eksik kalan tarama geçmek yerine hata verir
)

result = guard.scan(text)
payload = result.reapply(Action.TOKENIZE)   # açıkta `warn` değeri kalmaz
answer = call_llm(payload.sanitized_text)
print(payload.restore(answer))
```

- Tuzu, `token_map`'i ve geri konmuş metni kendi tarafınızda tutun: loglardan,
  izleme sistemlerinden ve sağlayıcının erişiminden uzakta. Log'a
  `result.redacted()` yazın.
- Geri istemeyeceğiniz değerler için `redact` kullanın.
- [kvkk preset'ini](presets.md) kendi verinizle gözden geçirin. Neyi
  kapsamadığı orada yazıyor.
