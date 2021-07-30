# AUTOMATED HOUSE PRICE PREDICTION ML

   ![Housing 2-](https://user-images.githubusercontent.com/73841520/127619877-9e16c198-5ae7-44b5-a5f5-308fc07d76d8.jpg)

# İş Problemi
* Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak, farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi gerçekleştirilmek istenmektedir.

# Veri Seti Hikayesi
* Ames, Lowa’daki konut evlerinden oluşan bu veriseti içerisinde 79 açıklayıcı değişken bulunduruyor.
* Veri seti bir kaggle yarışmasına ait olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır.
* Test veri setinde ev fiyatları boş bırakılmış olup, bu değerleri bizim tahmin etmemiz beklenmektedir.

# DEĞİŞKENLER
## Kategorik Değişkenler:
*  'MSZoning',         -Satışın genel imar sınıflandırmasını belirler.
*  'Street',           -Mülke bağlı caddenin doğrusal ayakları (Grvl: Çakıl, Pave: Döşeli)
*  'Alley',            -Mülke geçit erişimi türü (Grvl: Çakıl, Pave: Döşeli, NA: Geçit erişimi yok)
*  'LotShape',         -Mülkün genel şekli (Reg: Düzenli, IR1: Biraz düzensiz, IR2:Orta derecede düzzensiz, IR:Düzensiz)
*  'LandContour',      -Mülkün düzlüğü (Lvl: Yere yakın,Bnk: Sokak seviyesinden yüksekte, HLS: Eğimli, Low: Düşük seviyede)
*  'Utilities',        -Kullanılabilir yardımcı program türleri (gaz, elektrik vs)
*  'LotConfig',        -arsa sınıflandırması (Inside: iç kısımda, Corner: köşede, CulDSac: çıkmaz sokak, FR2: iki cepheli, FR3: üç cepheli)
*  'LandSlope',        -mülkün eğimi (Gtl: Hafif, Mod: Orta, Sev: Şiddetli)
*  'Condition1',       -Çeşitli koşullara yakınlık
*  'Condition2',       -Çeşitli koşullara yakınlık (birden fazla varsa)
*  'BldgType',         -konut tipi
*  'HouseStyle',       -konut tarzı
*  'RoofStyle',        -çatı tipi
*  'RoofMatl',         -çatı malzemesi
*  'Exterior1st',      -evin dış kaplaması
*  'Exterior2nd',      -Evin dış kaplaması (birden fazla malzeme varsa)
*  'MasVnrType',       -Duvar kaplama tipi
*  'ExterQual',        -Dış cephedeki malzemenin kalitesini değerlendirir
*  'ExterCond',        -Dış cephedeki malzemenin mevcut durumunu değerlendirir
*  'Foundation',       -yapının türü
*  'BsmtQual',         -Bodrumun yüksekliğini değerlendirir
*  'BsmtCond',         -Bodrumun genel durumunu değerlendirir
*  'BsmtExposure',     -bahçe seviyesindeki duvarları ifade eder
*  'BsmtFinType1',     -Bodrum bitmiş alanın değerlendirmesi
*  'BsmtFinType2',     -Bodrum bitmiş alanın değerlendirmesi (birden fazla tip varsa)
*  'Heating',          -Isıtma türü
*  'HeatingQC',        -Isıtma kalitesi ve durumu
*  'CentralAir',       -Merkezi klima
*  'Electrical',       -Elektrik sistemi
*  'KitchenQual',      -mutfak kalitesi
*  'Functional',       -Ev işlevselliği (Kesintiler garanti edilmediği sürece tipik olduğunu varsayın)
*  'FireplaceQu',      -şömine kalitesi
*  'GarageType',       -garaj konumu
*  'GarageFinish',     -garaj iç dekorasyonu
*  'GarageQual',       -garaj kalitesi
*  'GarageCond',       -garaj durumu
*  'PavedDrive',       -asfalt yol
*  'PoolQC',           -havuz kalitesi
*  'Fence',            -çit kalitesi
*  'MiscFeature',      -Diğer kategorilerde kapsanmayan çeşitli özellikler
*  'SaleType',         -satış tipi
*  'SaleCondition',    -satış durumu
*  'OverallCond',      -Evin genel durumunu değerlendirir
*  'BsmtFullBath',     -Bodrum katı banyoları
*  'BsmtHalfBath',     -Bodrum küçük banyoları
*  'FullBath',         -zemin üzerindeki katlardaki banyolar
*  'HalfBath',         -zemin üzerindeki küçük banyolar
*  'BedroomAbvGr',     -zemin üzerindeki katlardaki toplam yatak odası sayısı (bodrum katı yatak odaları dahil DEĞİLDİR)
*  'KitchenAbvGr',     -zemin üzerindeki katlardaki toplam mutfak sayısı
*  'Fireplaces',       -şömine sayısı
*  'GarageCars',       -Araba kapasiteli garajın büyüklüğü
*  'YrSold']           -Konutun satıldığı yıl

## Numerik Değişkenler:
*  'Id',
*  'MSSubClass',       -satılacak konut türünü tanımlar (dublex ve aile için tek katlı vs..)
*  'LotFrontage',      -mülke bağlı cadde sayısı
*  'LotArea',          -metrekare cinsinden arsa büyüklüğü
*  'OverallQual',      -evin genel malzemesinin ve son halinin değerlendirilmesi
*  'YearBuilt',        -evin yapım tarihi
*  'YearRemodAdd',     -evin tadilat gördüğü tarih (tadilat görmemişse, evin yapım tarihi ile aynıdır.)
*  'MasVnrArea',       -Metrekare cinsinden duvar kaplama alanı
*  'BsmtFinSF1',       -Tip 1 bitmiş metrekare
*  'BsmtFinSF2',       -Tip 2 bitmiş metrekare
*  'BsmtUnfSF',        -Bitmemiş metrekarelik bodrum alanı
*  'TotalBsmtSF',      -Bodrum alanının toplam metrekaresi
*  '1stFlrSF',         -Birinci kat metrekaresi
*  '2ndFlrSF',         -İkinci kat metrekaresi
*  'LowQualFinSF',     -Düşük kaliteli bitmiş fit kare (tüm katlar)
*  'GrLivArea',        -zemin üzerindeki toplam yaşam alanı metrekaresi
*  'TotRmsAbvGrd',     -zemin üzerindeki toplam oda sayısı (banyolar dahil değil)
*  'GarageYrBlt',      -Garajın yapıldığı yıl
*  'GarageArea',       -garajın metrekare cinsinden büyüklüğü
*  'WoodDeckSF',       -Metrekare cinsinden ahşap güverte alanı
*  'OpenPorchSF',      -Metrekare cinsinden açık sundurma alanı
*  'EnclosedPorch',    -Metrekare cinsinden kapalı sundurma alanı
*  '3SsnPorch',        -Metrekare olarak üç mevsim sundurma alanı
*  'ScreenPorch',      -Metrekare cinsinden cam kaplı sundurma alanı
*  'PoolArea',         -Havuz alanı metrekaresi
*  'MiscVal',          -çeşitli diğer özelliklerin getirdiği değer (dolar)
*  'MoSold',           -konutun satıldığı ay
*  'SalePrice']        -mülkün satış değeri (dolar)

# Kardinal Değişkenler:
* 'Neighborhood'
