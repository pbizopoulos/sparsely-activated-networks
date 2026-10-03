{ pkgs, ... }:
let
  pname = baseNameOf ./.;
  prmInstall = "";
  runtimeDeps = [ ];
  site = pkgs.runCommand "${pname}-site" { } ''
    mkdir -p "$out"
    cp ${./index.html} "$out/index.html"
    for asset in script.js style.css; do
      if [ -f ${./.}/"$asset" ]; then
        cp ${./.}/"$asset" "$out/$asset"
      fi
    done
    if [ -d ${./.}/prm ]; then
      cp -R ${./.}/prm "$out/prm"
      chmod -R u+w "$out/prm"
    fi
    ${prmInstall}
  '';
in
pkgs.writeShellApplication {
  meta.description = "An HTML, CSS, and JavaScript template package.";
  name = pname;
  runtimeInputs = runtimeDeps ++ [ pkgs.http-server ];
  text = ''
    exec http-server ${site} "$@"
  '';
}
