# Synthesizes Windows SAPI speech for the keywords and many other words (16 kHz mono WAV).
# Usage: powershell -File scripts/gen_tts.ps1 <out_dir>
# Output name: <word>__<voice>_<rate>.wav ; spaces in the word become '_'.
param([string]$Out = "data/tts")
Add-Type -AssemblyName System.Speech
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$keywords = 'left','right','up','down','go','stop'
$others = @'
hello hi hey okay testing test computer yellow morning night good bad yes no maybe please thanks thank sorry
low below follow row open close pause play dawn town gown brown frown clown crown drown now how cow wow
bow show slow snow flow glow grow know throw blow window yellow pillow follow hollow shadow
lift light right-wing write bright night fight sight tight might height flight
uplift upon update upset upper cup pup sup
done gone gun fun sun run won ton none one two three four five six seven eight nine ten zero
tree free three knee see sea bee tea key me we she he be
bed red head said dead ted fed bread thread
cat bat hat mat rat sat fat flat that what
dog log fog hog jog frog blog
bird word heard third nerd
house mouse blouse spouse
water paper table chair phone music video sound voice speech word light dark
music volume next previous slide forward back again repeat start begin end finish
red blue green black white orange purple pink gray
monday tuesday friday sunday today tomorrow yesterday
apple banana mango grape lemon
mother father brother sister friend teacher student
happy sad angry tired hungry thirsty
big small tall short long wide thin
hot cold warm cool wet dry
fast slow quick quiet loud soft hard easy
city country river mountain ocean forest desert
one more less most some any many few all every
about above across after against along among around before behind beside between beyond
because but and or if then than though although while until
can could will would shall should may might must
I you he she it we they me him her us them
is are was were be been being have has had do does did
hmm uh um ah oh wow ugh huh
how are you what is your name nice to meet you
'@ -split '\s+' | Where-Object { $_ } | Select-Object -Unique
$phrases = 'hello hello hello','hi there','good morning','how are you','what is that','one two three','are you ready','see you later','nice to meet you','thank you very much'
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono)
$words = @($keywords) + @($others) + @($phrases)
foreach ($v in 'Microsoft David Desktop','Microsoft Zira Desktop') {
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $s.SelectVoice($v)
  foreach ($r in -3,-1,0,1,3) {
    $s.Rate = $r
    foreach ($w in $words) {
      $name = ($w -replace '[^a-zA-Z]+','_') + '__' + $v.Split(' ')[1] + "_$r.wav"
      $s.SetOutputToWaveFile((Join-Path $Out $name), $fmt)
      $s.Speak($w)
      $s.SetOutputToNull()
    }
  }
}
