{
  pkgs ? import <nixpkgs> { },
}:
pkgs.writeShellApplication rec {
  meta.description = "An HTML, CSS, and JavaScript template package.";
  name = baseNameOf ./.;
  runtimeInputs = [ pkgs.http-server ];
  text = ''
    exec ${pkgs.http-server}/bin/http-server ${./.} "$@"
  '';
}
