{ pkgs, ... }:
let
  pname = baseNameOf ./.;
  runtimeDeps = [ ];
in
pkgs.writeShellApplication {
  meta.description = "An HTML, CSS, and JavaScript template package.";
  name = pname;
  runtimeInputs = runtimeDeps ++ [ pkgs.http-server ];
  text = ''
    exec http-server ${./.} "$@"
  '';
}
