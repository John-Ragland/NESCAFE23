"""Ray identifier utilities for NESCAFE23."""


def parse_ray_id(ray_id: str) -> tuple:
    """
    Convert ray_id into pygenray compatible format.

    Parameters
    ----------
    ray_id : str
        Ray identifier string (e.g., '-7', '+10b').

    Returns
    -------
    ray_id_pr : str
        Converted ray identifier in pygenray format (e.g., '-7.0', '+10.0b').
    (sign, number, suffix) : tuple
        Parsed components of the ray identifier.
    """

    # remove upper array suffix if present
    if ray_id.endswith("u"):
        ray_id = ray_id[:-1]

    sign = ray_id[0]  # '+' or '-'
    if ray_id.endswith("b"):
        number_str = ray_id[1:-1]  # digits between sign and 'b'
        suffix = "b"
    else:
        number_str = ray_id[1:]  # digits after sign
        suffix = ""

    number = int(number_str)

    ray_id_pr = f"-{number}.0{suffix}" if sign == "-" else f"{number}.0{suffix}"

    return ray_id_pr, (sign, number, suffix)


def n_botts_ray_id(ray_ids: list) -> dict:
    """
    Compute the number of bottom bounces for each ray ID.

    Parameters
    ----------
    ray_ids : list of str
        Ray identifier strings (e.g., ['-7', '+10b']).

    Returns
    -------
    n_botts : dict
        Dictionary with keys of ray_ids and values are the number of bottom
        bounces for each ray_id.
    """
    n_botts = []
    for ray_id in ray_ids:
        _, (ray_id_sign, ray_id_num, _) = parse_ray_id(ray_id)

        if ray_id_num == 0:
            n_botts.append(0)
        elif ray_id_num % 2 == 0:
            n_botts.append(ray_id_num / 2)
        else:
            if ray_id_sign == "+":
                n_botts.append((ray_id_num - 1) / 2)
            else:
                n_botts.append((ray_id_num + 1) / 2)

    return dict(zip(ray_ids, n_botts, strict=False))
