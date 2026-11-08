Output assets/demo.gif

Set Shell "bash"
Set FontFamily "JetBrains Mono"
Set FontSize 20
Set Width 1200
Set Height 675
Set Padding 24
Set Framerate 30
Set WaitTimeout 180s
Set WaitPattern /^\$$/
Set Theme { "name": "SynthStatsDark", "background": "#0D0F17", "foreground": "#E9EBF5", "cursor": "#ABA6F2", "selection": "#242839", "black": "#151925", "red": "#D41950", "green": "#0096D0", "yellow": "#ABA6F2", "blue": "#0072B2", "magenta": "#5C56A5", "cyan": "#0096D0", "white": "#E9EBF5", "brightBlack": "#A0A6BC", "brightRed": "#F03268", "brightGreen": "#0096D0", "brightYellow": "#ABA6F2", "brightBlue": "#0096D0", "brightMagenta": "#ABA6F2", "brightCyan": "#0096D0", "brightWhite": "#FBFBFD" }

Hide
Type "export PS1='$ ' && export PIXI_FROZEN=true && export PIXI_NO_INSTALL=true && clear"
Enter
Wait
Type "pixi run -e dev psrh --help >/dev/null 2>&1"
Enter
Wait
Type "clear"
Enter
Wait
Show

Type "cat configs/data/bernoulli_1d.yaml"
Sleep 400ms
Enter
Wait+Screen /bernoulli|split|n_/
Sleep 2s

Type "pixi run -e dev psrh data generate --config configs/data/bernoulli_1d.yaml --seed 0"
Sleep 400ms
Enter
Wait+Screen /dataset_id/
Sleep 5s
