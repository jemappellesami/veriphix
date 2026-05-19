OPENQASM 2.0;
include "qelib1.inc";
qreg q104[4];
cx q104[0],q104[1];
cx q104[2],q104[3];
rz(pi) q104[1];
cx q104[2],q104[1];
cx q104[1],q104[0];
